import sys
import ujson as json
from collections import defaultdict

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from rewards.orm import ground_truth_reward, rule_score_reward, length_reward, web_rule_score_reward, web_response_score_reward
from rewards.prm import format_reward
from util.message import Message, OUTPUT_ROLES

FIELDS = ["title", "price", "service", "sku & attrs"]

searcher = LuceneSearcher("indexes")


def load_rollout_outputs(config: dict) -> dict:
    rollout_outputs = dict()
    cnt = 0
    with open(config["rollout_file"], "r") as fin:
        for line in tqdm(fin, desc="Load roll out outputs: "):
            if cnt>500:
                break
            jsonobj = json.loads(line.strip())
            query = jsonobj[0]["extra_info"]["query"]
            rollout_outputs[query] = jsonobj
            cnt +=1
    return rollout_outputs


def load_synthesize_rewards(config: dict) -> dict:
    synthesize_rewards = dict()
    with open(config["synthesize_file"], "r") as fin:
        for line in tqdm(fin, desc="Load synthesize rewards: "):
            jsonobj = json.loads(line.strip())
            query = jsonobj["query"]
            reward = jsonobj["reward"]
            synthesize_rewards[query] = reward
    return synthesize_rewards


def load_synthesize_vouchers(config: dict) -> dict:
    synthesize_vouchers = dict()
    with open(config["synthesize_file"], "r") as fin:
        for line in tqdm(fin, desc="Load synthesize vouchers: "):
            jsonobj = json.loads(line.strip())
            query = jsonobj["query"]
            voucher = jsonobj.get("voucher")
            if query and voucher:
                synthesize_vouchers[query] = voucher
    return synthesize_vouchers


def extract_recommed_product(output: list[dict]):
    product_ids = ""
    if not output:
        return product_ids

    for step in output:
        message = step["completion"]["message"]
        if message and "tool_call" in message and message["tool_call"]:
            for commend in message["tool_call"]:
                if commend["name"] == "recommend_product":
                    product_ids = commend["parameters"].get("product_ids", "")
    if not isinstance(product_ids, str):
        return ""
    return product_ids


def set_eval_score(product: dict, score: dict, reward: dict):
    score["product"] += 1

    score["gt"] += ground_truth_reward(product, reward)

    rule_score, total_counter, hit_counter = rule_score_reward(product, reward)
    score["rule"] += rule_score
    for field in FIELDS:
        score[field] += hit_counter.get(field, 0) / total_counter.get(field, 0) if total_counter.get(field, 0) > 0 else 1


def eval_product(score: dict, output: list[dict], reward: dict):
    product_ids = extract_recommed_product(output)
    product_id = product_ids.split(",")[0]

    doc = searcher.doc(product_id)
    if not doc:
        return
    product = json.loads(doc.raw())["product"]

    set_eval_score(product, score, reward)


def eval_shop(score: dict, output: list[dict], reward: list[dict]):
    num_hits = 0
    shop_ids = set()
    product_ids = extract_recommed_product(output)
    product_id_list = product_ids.split(",")
    for i, sub_reward in enumerate(reward):
        if i >= len(product_id_list):
            continue
        product_id = product_id_list[i]

        doc = searcher.doc(product_id)
        if not doc:
            continue
        product = json.loads(doc.raw())["product"]

        set_eval_score(product, score, sub_reward)
        num_hits += 1
        shop_ids.add(product["shop_id"])

    score["product"] /= len(reward)
    score["gt"] /= len(reward)
    score["rule"] /= len(reward)
    for field in FIELDS:
        score[field] /= len(reward)
    score["shop"] = 1 if num_hits == len(reward) and len(shop_ids) == 1 else 0


def eval_voucher(score: dict, output: list[dict], reward: list[dict], voucher: dict):
    num_hits = 0
    total_price = 0
    shop_ids = set()
    product_ids = extract_recommed_product(output)
    product_id_list = product_ids.split(",")
    for i, sub_reward in enumerate(reward):
        if i >= len(product_id_list):
            continue
        product_id = product_id_list[i]

        doc = searcher.doc(product_id)
        if not doc:
            continue
        product = json.loads(doc.raw())["product"]

        set_eval_score(product, score, sub_reward)
        num_hits += 1
        total_price += product["price"]
        shop_ids.add(product["shop_id"])

    budget_match = 0
    if num_hits == len(reward):
        if total_price <= voucher["budget"]:
            budget_match = 1
        elif voucher["voucher_type"] == "platform" or (voucher["voucher_type"] == "shop" and len(shop_ids) == 1):
            if total_price >= voucher["threshold"]:
                if voucher["discount_type"] == "fixed":
                    total_price_after_discount = total_price - voucher["face_value"]
                elif voucher["discount_type"] == "percentage":
                    total_price_after_discount = max(total_price * (1 - voucher["discount"]), total_price - voucher["cap"])
                else:
                    raise Exception(f"Invalid voucher discount type: {voucher['discount_type']}")
                budget_match = 1 if total_price_after_discount <= voucher["budget"] else 0

    score["product"] /= len(reward)
    score["gt"] /= len(reward)
    score["rule"] /= len(reward)
    for field in FIELDS:
        score[field] /= len(reward)
    score["budget"] = budget_match


def evaluate(config: dict):
    if config["task"] == "web":
        eval_web(config)
        return
    task = config["task"]
    rollout_outputs = load_rollout_outputs(config)
    synthesize_rewards = load_synthesize_rewards(config)
    synthesize_vouchers = load_synthesize_vouchers(config)
    if "ablation_react" in config["rollout_file"]:
        mode = "no think"
    else:
        mode = "think"

    # Calculate reward
    results = dict()
    for query in tqdm(rollout_outputs.keys(), desc="Calculate reward: "):
        if query not in synthesize_rewards:
            continue
        output = rollout_outputs[query]
        reward = synthesize_rewards[query]
        voucher = synthesize_vouchers.get(query)

        score = defaultdict(float)

        # length score
        length_score = length_reward(output)
        score["length"] = length_score

        # format score
        format_score = 0
        if config["model_config"]["model"] != "human":
            for step in output:
                message = Message.from_dict(step["completion"]["message"])
                completion = message.to_string(OUTPUT_ROLES)
                format_score += format_reward(completion) if mode == "think" else format_reward(completion, ["tool_call"])
        format_score = format_score / len(output) if output else 0
        score["format"] = format_score

        # eval score
        if task == "product":
            eval_product(score, output, reward)
        elif task == "shop":
            eval_shop(score, output, reward)
        elif task == "voucher":
            eval_voucher(score, output, reward, voucher)
        else:
            raise Exception(f"Invalid task: {task}")

        results[query] = score

    print(f"Model `{config['model_config']['model']}` Rollout `{len(results)}` cases:")

    print("--- Metrics ---")

    gt_rate = len([v for v in results.values() if v["gt"] >= 1]) / len(results)
    print(f"The gt rate is {gt_rate:.3f}")

    if task == "product":
        success_rate = len([v for v in results.values() if v["rule"] >= 1]) / len(results)
    elif task == "shop":
        success_rate = len([v for v in results.values() if v["rule"] >= 1 and v["shop"] >= 1]) / len(results)
    elif task == "voucher":
        success_rate = len([v for v in results.values() if v["rule"] >= 1 and v["budget"] >= 1]) / len(results)
    else:
        raise Exception(f"Invalid task: {task}")
    print(f"The success rate is {success_rate:.3f}")

    print("--- Details ---")

    format_score = sum(v["format"] for v in results.values()) / len(results)
    print(f"The format score is {format_score:.3f}")

    recommend_product_score = sum([v["product"] for v in results.values()]) / len(results)
    print(f"The recommend product score is {recommend_product_score:.3f}")

    for field in FIELDS:
        field_match_score = sum([v[field] for v in results.values()]) / len(results)
        print(f"The {field} match score is {field_match_score:.3f}")

    rule_match_score = sum([v["rule"] for v in results.values()]) / len(results)
    print(f"The rule match score is {rule_match_score:.3f}")
    if task == "shop":
        shop_match_score = sum([v["shop"] for v in results.values()]) / len(results)
        print(f"The shop match score(rate) is {shop_match_score:.3f}")
    elif task == "voucher":
        budget_match_score = sum([v["budget"] for v in results.values()]) / len(results)
        print(f"The budget match score(rate) is {budget_match_score:.3f}")


def evaluate_voucher(config: dict):
    pass



def load_synthesize_web_rewards(config: dict) -> dict:
    synthesize_rewards = dict()
    with open(config["synthesize_file"], "r") as fin:
        for line in tqdm(fin, desc="Load synthesize web rewards: "):
            jsonobj = json.loads(line.strip())
            query = jsonobj["query"]
            reward = jsonobj["reward"]
            kw = jsonobj["Knowledge_Attribute"]
            synthesize_rewards[query] = reward, str(kw)
    return synthesize_rewards
  
    
def eval_web(config: dict):
    rollout_outputs = load_rollout_outputs(config)
    synthesize_rewards = load_synthesize_web_rewards(config)
    # Calculate reward
    results = dict()
    for query in tqdm(rollout_outputs.keys(), desc="Calculate reward: "):
        if query not in synthesize_rewards:
            continue
        output = rollout_outputs[query]
        reward, kw = synthesize_rewards[query]
        reward['key_attribute'] = kw
        score = {
            "gt": 0,
            "rule": 0,
            "length": length_reward(output),
            "format": (
                sum(format_reward(step["completion"]["content"]) for step in output)
                / len(output)
                if output
                else 0
            ),
        }
        results[query] = score

        response = "\n".join([item['completion']['message'].get('response', '') for item in output])

        product_ids = extract_recommed_product(output)
        product_id = product_ids.split(",")[0]
        score["have_recommend"] = 1 if product_id else 0
        score["gt"] = 1 if reward['product_id'] in product_id else 0
        doc = searcher.doc(product_id)
        if doc:
            product = json.loads(doc.raw())["product"]
            score["kw"], score["title"] = web_rule_score_reward(product, reward)
        else:
            score["kw"], score["title"] = 0, 0
        score["response"] = max(score["kw"], web_response_score_reward(response, kw))
        score['rule'] = (score['kw'] + score['title'])/2
        results[query] = score
    
    # report scores
    gt_pass_at_1 = len([v for v in results.values() if v["gt"] >= 1]) / len(results)
    rule_pass_at_1 = len([v for v in results.values() if v["rule"] >= 1]) / len(results)
    rule_score = sum([v["rule"] for v in results.values()]) / len(results)
    # detail scores
    title_score = sum([v["title"] for v in results.values()]) / len(results)
    kw_score = sum([v["kw"] for v in results.values()]) / len(results)
    response_score = sum([v["response"] for v in results.values()]) / len(results)
    have_recommend_score = sum([v["have_recommend"] for v in results.values()]) / len(results)
    

    # format scores
    avg_lrs = sum(v["length"] for v in results.values()) / len(results)
    avg_format_score = sum(v["format"] for v in results.values()) / len(results)

    print(f"Model `{config['model_config']['model']}` Rollout `{len(results)}` cases:")
    print("="*50+ "report scores" + "="*50)
    print(f"The ground-truth pass@1 is {gt_pass_at_1:.3f}.")
    print(f"The success rate is {rule_pass_at_1:.3f}.")
    # print(f"The matching score is {rule_score:.3f}.")
    print(f"The kw score is {kw_score:.3f}.")
    print(f"The title score is {title_score:.3f}.")

    print("="*50+ "detail scores" + "="*50)
    print(f"The have recommend score is {have_recommend_score:.3f}.")
    # print(f"The response reward score is {response_score:.3f}.")

    print("="*50+ "format scores" + "="*50)
    print(f"The format reward score is {avg_format_score:.3f}.")
    print(f"The length reward score is {avg_lrs:.3f}.")



if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    evaluate(config)
