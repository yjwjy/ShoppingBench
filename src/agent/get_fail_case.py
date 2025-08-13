import sys
import ujson as json
from collections import defaultdict

import tiktoken
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from run_evaluate import (
    load_rollout_outputs,
    load_synthesize_rewards,
    load_synthesize_vouchers,
    load_synthesize_web_rewards,
    extract_recommed_product,
    eval_product,
    eval_shop,
    eval_voucher
)

from rewards.orm import length_reward, web_rule_score_reward, web_response_score_reward
from rewards.prm import format_reward
from util.message import Message, OUTPUT_ROLES
import random

random.seed(42)

searcher = LuceneSearcher("indexes")
enc = tiktoken.encoding_for_model("gpt-4o")

def eval_web(score, output, reward, kw):
    reward['key_attribute'] = kw
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

    return score


def reject_sample(config: dict):
    task = config["task"]
    if task == "web":
        synthesize_rewards = load_synthesize_web_rewards(config)
    else:
        synthesize_rewards = load_synthesize_rewards(config)
    vouchers = load_synthesize_vouchers(config)
    # Randomly sample 50 queries
    sampled_queries = random.sample(list(synthesize_rewards.keys()), min(50, len(synthesize_rewards)))
    synthesize_rewards = {q: synthesize_rewards[q] for q in sampled_queries}
    outputs_list = []
    mode_list = []

    for rollout_file in config["rollout_files"]:
        outputs = load_rollout_outputs({"rollout_file": rollout_file})
        outputs_list.append(outputs)
        mode_list.append("think")
    total = set()
    success_ns = set()
    samples = []
    fail_samples = []
    success_scores = []
    fail_scores = []
    total_scores = []
    for query in tqdm(synthesize_rewards.keys()):
        max_output = None
        max_score = None
        max_length_score = 0

        for outputs in outputs_list:
            if query not in outputs:
                continue
            total.add(query)

            output = outputs[query]
            reward = synthesize_rewards[query]
            voucher = vouchers.get(query)

            length_score = length_reward(output)
            score = defaultdict(float)

            if task == "web":
                reward, kw = synthesize_rewards[query]
                score = eval_web(score, output, reward, kw)
                output[-1]['score'] = score
                if score["rule"] < 1:
                    continue
            elif task == "product":
                eval_product(score, output, reward)
                if score["rule"] < 1:
                    continue
            elif task == "shop":
                eval_shop(score, output, reward)
                if score["rule"] < 1 or score["shop"] < 1:
                    continue
            elif task == "voucher":
                eval_voucher(score, output, reward, voucher)
                if score["rule"] < 1 or score["budget"] < 1:
                    continue
            else:
                raise Exception(f"Invalid task: {task}")

            if length_score > max_length_score:
                max_output = output
                max_score = score
                max_length_score = length_score

        if not max_output:
            fail_samples.append(output)
            fail_scores.append(score)
            total_scores.append(score)
            continue
        
        success_ns.add(query)

        samples.append(max_output)
        success_scores.append(max_score)
        total_scores.append(max_score)

    # print(len(samples))
    with open(config["human_anatation_file"], "w") as fout:
        for item in samples:
            fout.write(f"{json.dumps(item)}\n")
    
    with open(config["fail_data"], "w") as fout:
        for item in fail_samples:
            fout.write(f"{json.dumps(item)}\n")
    
    return samples, fail_samples, total_scores


if __name__ == "__main__":
    import os
    data_simpleqa_dir = "data/simpleqa"
    simpleqa_rollout_files = [
        os.path.join(data_simpleqa_dir, f)
        for f in os.listdir(data_simpleqa_dir)
        if os.path.isfile(os.path.join(data_simpleqa_dir, f))
    ]
    data_product_dir = "data/product"
    product_rollout_files = [
        os.path.join(data_product_dir, f)
        for f in os.listdir(data_product_dir)
        if os.path.isfile(os.path.join(data_product_dir, f))
    ]
    data_shop_dir = "data/shop"
    shop_rollout_files = [
        os.path.join(data_shop_dir, f)
        for f in os.listdir(data_shop_dir)
        if os.path.isfile(os.path.join(data_shop_dir, f))
    ]
    data_voucher_dir = "data/voucher"
    voucher_rollout_files = [
        os.path.join(data_voucher_dir, f)
        for f in os.listdir(data_voucher_dir)
        if os.path.isfile(os.path.join(data_voucher_dir, f))
    ]

    simpleqa_rollout_files = [item for item in simpleqa_rollout_files if "gpt-4.1" in item]
    print(simpleqa_rollout_files)
    product_rollout_files = [item for item in product_rollout_files if "gpt-4.1" in item]
    print(product_rollout_files)
    shop_rollout_files = [item for item in shop_rollout_files if "gpt-4.1" in item]
    print(shop_rollout_files)
    voucher_rollout_files = [item for item in voucher_rollout_files if "gpt-4.1" in item]
    print(voucher_rollout_files)
    simpelqa_config = {
        "task": "web",
        "synthesize_file": "data/synthesize_web_simpleqa_test.jsonl",
        "rollout_files": simpleqa_rollout_files,
        "human_anatation_file": "data/example/rollout_simpleqa_web_human.json",
        "fail_data" : "src/agent/annotate/gpt41_simpleqa_failure.json"
    }
    product_config = {
        "task": "product",
        "synthesize_file": "data/synthesize_product_test.jsonl",
        "rollout_files": product_rollout_files,
        "human_anatation_file": "data/example/rollout_product_human.json",
        "fail_data" : "src/agent/annotate/gpt41_product_failure.json"
    }
    shop_config = {
        "task": "shop",
        "synthesize_file": "data/synthesize_shop_test.jsonl",
        "rollout_files": shop_rollout_files,
        "human_anatation_file": "data/example/rollout_shop_human.json",
        "fail_data" : "src/agent/annotate/gpt41_shop_failure.json"
    }
    voucher_config = {
        "task": "voucher",
        "synthesize_file": "data/synthesize_voucher_test.jsonl",
        "rollout_files": voucher_rollout_files,
        "human_anatation_file": "data/example/rollout_voucher_human.json",
        "fail_data" : "src/agent/annotate/gpt41_voucher_failure.json"
    }
    # simpelqa_config = {
    #     "task": "web",
    #     "synthesize_file": "data/synthesize_web_simpleqa_test.jsonl",
    #     "rollout_files": simpleqa_rollout_files,
    #     "human_anatation_file": "data/rollout_simpleqa_web_human.json",
    #     "fail_data" : "src/agent/annotate/all_model_simpleqa_failure.json"
    # }
    # product_config = {
    #     "task": "product",
    #     "synthesize_file": "data/synthesize_product_test.jsonl",
    #     "rollout_files": product_rollout_files,
    #     "human_anatation_file": "data/rollout_product_human.json",
    #     "fail_data" : "src/agent/annotate/all_model_product_failure.json"
    # }
    # shop_config = {
    #     "task": "shop",
    #     "synthesize_file": "data/synthesize_shop_test.jsonl",
    #     "rollout_files": shop_rollout_files,
    #     "human_anatation_file": "data/rollout_shop_human.json",
    #     "fail_data" : "src/agent/annotate/all_model_shop_failure.json"
    # }
    # voucher_config = {
    #     "task": "voucher",
    #     "synthesize_file": "data/synthesize_voucher_test.jsonl",
    #     "rollout_files": voucher_rollout_files,
    #     "human_anatation_file": "data/rollout_voucher_human.json",
    #     "fail_data" : "src/agent/annotate/all_model_voucher_failure.json"
    # }

    samples, fail_samples, total_scores = reject_sample(simpelqa_config)
    print(len(samples))
    print(len(fail_samples))
    print(len(total_scores))
    rule_pass_at_1 = len([v for v in total_scores if v["rule"] >= 1]) / len(total_scores)
    # detail scores
    title_score = sum([v["title"] for v in total_scores]) / len(total_scores)
    kw_score = sum([v["kw"] for v in total_scores]) / len(total_scores)
    print(f"The success rate is {rule_pass_at_1:.3f}.")
    print(f"The kw score is {kw_score:.3f}.")
    print(f"The title score is {title_score:.3f}.")


    samples, fail_samples, total_scores = reject_sample(product_config)
    print(len(samples))
    print(len(fail_samples))
    print(len(total_scores))
    rule_pass_at_1 = len([v for v in total_scores if v["rule"] >= 1]) / len(total_scores)
    # detail scores
    rule_match_score = sum([v["rule"] for v in total_scores]) / len(total_scores)
    print(f"The success rate is {rule_pass_at_1:.3f}.")
    print(f"The rule match score is {rule_match_score:.3f}.")


    samples, fail_samples, total_scores = reject_sample(shop_config)
    print(len(samples))
    print(len(fail_samples))
    print(len(total_scores))
    rule_pass_at_1 = len([v for v in total_scores if v["rule"] >= 1 and v["shop"] >= 1]) / len(total_scores)
    # detail scores
    rule_match_score = sum([v["rule"] for v in total_scores]) / len(total_scores)
    shop_match_score = sum([v["shop"] for v in total_scores]) / len(total_scores)
    print(f"The success rate is {rule_pass_at_1:.3f}.")
    print(f"The rule match score is {rule_match_score:.3f}.")
    print(f"The shop score is {shop_match_score:.3f}.")


    samples, fail_samples, total_scores = reject_sample(voucher_config)
    print(len(samples))
    print(len(fail_samples))
    print(len(total_scores))
    rule_pass_at_1 = len([v for v in total_scores if v["rule"] >= 1 and v["budget"] >= 1]) / len(total_scores)
    # detail scores
    rule_match_score = sum([v["rule"] for v in total_scores]) / len(total_scores)
    budget_match_score = sum([v["budget"] for v in total_scores]) / len(total_scores)
    print(f"The success rate is {rule_pass_at_1:.3f}.")
    print(f"The rule match score is {rule_match_score:.3f}.")
    print(f"The budget score is {budget_match_score:.3f}.")