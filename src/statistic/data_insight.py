import os
import sys
import ujson as json
from collections import Counter, defaultdict

import tiktoken
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher


searcher = LuceneSearcher("indexes")
enc = tiktoken.encoding_for_model("gpt-4o")


def synthesize_data_insight(synthesize_file):
    query_tokens = []
    product_set = set()
    product_num_counter = Counter()
    cate_counter = Counter()
    fields_set = set()
    fields_num_counter = Counter()
    price_mode_counter = Counter()

    total = int(os.popen(f"wc -l {synthesize_file}").read().strip().split(" ", 1)[0])
    with open(synthesize_file, "r") as fin:
        for line in tqdm(fin, total=total):
            jsonobj = json.loads(line.strip())
            query = jsonobj["query"]
            reward = jsonobj["reward"]
            if not isinstance(reward, list):
                reward = [reward]

            # query tokens
            query_tokens.append(len(enc.encode(query)))

            # product number
            product_num_counter[len(reward)] += 1

            for sub_reward in reward:
                product_id = sub_reward["product_id"]

                # product set
                product_set.add(product_id)

                # category
                product = json.loads(searcher.doc(product_id).raw())["product"]
                category = product["category"]
                cate_level1_name = category.split(" > ")[0]
                if cate_level1_name:
                    cate_counter[cate_level1_name] += 1
                else:
                    cate_counter["Other"] += 1

                if "_web_" in synthesize_file:
                    continue

                # fields set
                fields_num = 0
                attributes = sub_reward.get("attributes", [])
                for attr in attributes:
                    for k, vs in attr.items():
                        for v in vs:
                            fields_set.add((k, v))
                            fields_num += 1

                service = sub_reward.get("service", [])
                for serv in service:
                    fields_set.add(serv)
                    fields_num += 1

                sku_options = sub_reward.get("sku_options", [])
                for option in sku_options:
                    for k, v in option.items():
                        fields_set.add((k, v))
                        fields_num += 1

                # fields num
                fields_num_counter[fields_num] += 1

                # price mode
                price = sub_reward.get("price", [])
                for p in price:
                    for price_mode, price_range in p.items():
                        price_mode_counter[price_mode] += 1
                if len(price) == 0:
                    price_mode_counter["no"] += 1

    print(f"===== {synthesize_file} =====")
    index = 1

    query_tokens.sort()
    print(
        f"{index}. Synthetic queries tokens min/max/avg/med: {query_tokens[0]}/{query_tokens[-1]}/{sum(query_tokens) / len(query_tokens):.3f}/{query_tokens[len(query_tokens) // 2]}"
    )
    index += 1

    print(f"{index}. Include {len(product_set)} unique products")
    index += 1

    product_num_distribution = "\n".join(
        str(k) + "\t" + f"{v / sum(product_num_counter.values()):.4f}"
        for k, v in sorted(product_num_counter.items())
    )
    print(
        f"\n{index}. The number of products distribution:\n{product_num_distribution}"
    )
    index += 1

    cate_distribution = "\n".join(
        str(k) + "\t" + f"{v / sum(cate_counter.values()):.4f}"
        for k, v in cate_counter.most_common()
    )
    print(f"\n{index}. The category distribution:\n{cate_distribution}")
    index += 1

    print(f"\n{index}. Include {len(fields_set)} unique fields")
    index += 1

    fields_num_distribution = "\n".join(
        str(k) + "\t" + f"{v / sum(fields_num_counter.values()):.4f}"
        for k, v in sorted(fields_num_counter.items())
    )
    print(f"\n{index}. The number of fields distribution:\n{fields_num_distribution}")
    index += 1

    price_mode_distribution = "\n".join(
        str(k) + "\t" + f"{v / sum(price_mode_counter.values()):.4f}"
        for k, v in sorted(price_mode_counter.items())
    )
    print(f"\n{index}. The price mode distribution:\n{price_mode_distribution}")
    index += 1


def documents_data_insight(documents_file):
    product_set = set()
    cate_counter = Counter()
    fields_set = set()
    fields_num_counter = Counter()

    total = int(os.popen(f"wc -l {documents_file}").read().strip().split(" ", 1)[0])
    with open(documents_file, "r") as fin:
        for line in tqdm(fin, total=total):
            product = json.loads(line.strip())["product"]
            product_id = product["product_id"]

            # product set
            product_set.add(product_id)

            # category
            product = json.loads(searcher.doc(product_id).raw())["product"]
            category = product["category"]
            cate_level1_name = category.split(" > ")[0]
            if cate_level1_name:
                cate_counter[cate_level1_name] += 1
            else:
                cate_counter["Other"] += 1

            # fields set
            fields_num = 0
            attributes = product.get("attributes", {})
            for k, vs in attributes.items():
                for v in vs:
                    fields_set.add((k, v))
                    fields_num += 1

            service = product.get("service", [])
            for serv in service:
                fields_set.add(serv)
                fields_num += 1

            sku_options = product.get("sku_options", {})
            for _, option in sku_options.items():
                for k, v in option.items():
                    fields_set.add((k, v))
                    fields_num += 1

            # fields num
            if fields_num >= 25:
                fields_num = 25
            fields_num_counter[fields_num] += 1

    print(f"===== {documents_file} =====")

    print(f"1. Include {len(product_set)} unique products")

    cate_distribution = "\n".join(
        str(k) + "\t" + f"{v / sum(cate_counter.values()):.4f}"
        for k, v in cate_counter.most_common()
    )
    print(f"\n2. The category distribution:\n{cate_distribution}")

    print(f"\n3. Include {len(fields_set)} unique fields")

    fields_num_distribution = "\n".join(
        str(k) + "\t" + f"{v / sum(fields_num_counter.values()):.4f}"
        for k, v in sorted(fields_num_counter.items())
    )
    print(f"\n4. The number of fields distribution:\n{fields_num_distribution}")


def rollout_data_insight(rollout_file, test_size):
    steps = []
    step_tokens = defaultdict(list)
    search_queries = []
    page_turning = []
    search_in_shop = []
    view_product_information = []
    web_search = []

    total = int(os.popen(f"wc -l {rollout_file}").read().strip().split(" ", 1)[0])
    with open(rollout_file, "r") as fin:
        for i, line in enumerate(fin):
            if i >= test_size:
                break

            q_set = set()
            page_turning_cnt = 0
            search_in_shop_cnt = 0
            view_cnt = 0
            web_search_cnt = 0

            trejectory = json.loads(line.strip())
            steps.append(len(trejectory))

            for step in trejectory:
                prompt = step["prompt"]
                completion = step["completion"]
                extra_info = step["extra_info"]

                system_prompt = [x["content"] for x in prompt if x["role"] == "system"][
                    0
                ]
                user_prompt = [x["content"] for x in prompt if x["role"] == "user"][0]

                reasoning_content = completion["reasoning_content"]
                content = completion["content"]
                if reasoning_content:
                    assistant_response = (
                        f"<think>{reasoning_content}</think>\n{content}"
                    )
                else:
                    assistant_response = content

                # step tokens
                step_tokens["system"].append(len(enc.encode(system_prompt)))
                step_tokens["user"].append(len(enc.encode(user_prompt)))
                step_tokens["assistant"].append(len(enc.encode(assistant_response)))

                message = completion["message"]
                if "tool_call" in message:
                    for commend in message["tool_call"]:
                        if commend["name"] == "find_product":
                            q = commend["parameters"].get("q")
                            page = commend["parameters"].get("page")
                            shop_id = commend["parameters"].get("shop_id")

                            if q:
                                q_set.add(q)
                            if page and (isinstance(page, int) or page.isdigit()) and int(page) > 1:
                                page_turning_cnt += 1
                            if shop_id and shop_id.isdigit():
                                search_in_shop_cnt += 1
                        elif commend["name"] == "view_product_information":
                            view_cnt += 1
                        elif commend["name"] == "web_search":
                            web_search_cnt += 1

            search_queries.append(len(q_set))
            page_turning.append(page_turning_cnt)
            search_in_shop.append(search_in_shop_cnt)
            view_product_information.append(view_cnt)
            web_search.append(web_search_cnt)

    assert len(steps) == test_size
    assert len(search_queries) == test_size
    assert len(page_turning) == test_size
    assert len(search_in_shop) == test_size
    assert len(view_product_information) == test_size
    assert len(web_search) == test_size

    print(f"===== {rollout_file} =====")
    index = 1
    result = dict()

    steps.sort()
    print(
        f"{index}. Steps min/max/avg/med: {steps[0]}/{steps[-1]}/{sum(steps) / len(steps):.3f}/{steps[len(steps) // 2]}"
    )
    index += 1
    result["steps"] = sum(steps) / len(steps)

    for key, value in step_tokens.items():
        value.sort()
        print(f"{index}. {key} tokens per step min/max/avg/med: {value[0]}/{value[-1]}/{sum(value) / len(value):.3f}/{value[len(value) // 2]}")
        index += 1
        result[f"{key} tokens per step"] = sum(value) / len(value)

    search_queries.sort()
    print(
        f"{index}. Search Queries min/max/avg/med: {search_queries[0]}/{search_queries[-1]}/{sum(search_queries) / len(search_queries):.3f}/{search_queries[len(search_queries) // 2]}"
    )
    index += 1
    result["search queries"] = sum(search_queries) / len(search_queries)

    page_turning.sort()
    print(
        f"{index}. Page Turning min/max/avg/med: {page_turning[0]}/{page_turning[-1]}/{sum(page_turning) / len(page_turning):.3f}/{page_turning[len(page_turning) // 2]}"
    )
    index += 1
    result["page turning"] = sum(page_turning) / len(page_turning)

    search_in_shop.sort()
    print(
        f"{index}. Find product in specific shop min/max/avg/med: {search_in_shop[0]}/{search_in_shop[-1]}/{sum(search_in_shop) / len(search_in_shop):.3f}/{search_in_shop[len(search_in_shop) // 2]}"
    )
    index += 1
    result["find product in specific shop"] = sum(search_in_shop) / len(search_in_shop)

    view_product_information.sort()
    print(
        f"{index}. View product information min/max/avg/med: {view_product_information[0]}/{view_product_information[-1]}/{sum(view_product_information) / len(view_product_information):.3f}/{view_product_information[len(view_product_information) // 2]}"
    )
    index += 1
    result["view product information"] = sum(view_product_information) / len(view_product_information)

    web_search.sort()
    print(
        f"{index}. Web search min/max/avg/med: {web_search[0]}/{web_search[-1]}/{sum(web_search) / len(web_search):.3f}/{web_search[len(web_search) // 2]}"
    )
    index += 1
    result["web search"] = sum(web_search) / len(web_search)

    return result


if __name__ == "__main__":
    task = sys.argv[1]
    assert task in ["synthesize", "documents", "rollout"]

    if task == "synthesize":
        synthesize_files = [
            "data/synthesize_product.jsonl",
            "data/synthesize_shop.jsonl",
            "data/synthesize_voucher.jsonl",
            "data/synthesize_web_simpleqa.jsonl",
        ]
        for synthesize_file in synthesize_files:
            synthesize_data_insight(synthesize_file)
    elif task == "documents":
        documents_data_insight("resources/documents.jsonl")
    elif task == "rollout":
        rollout_files = []
        for task in ["product", "shop", "voucher", "web"]:
            task_result = dict()
            test_size = 250 
            if task == "web":
                test_size = 150
            for model in tqdm(
                [
                    "gpt-4.1",
                    "o3-mini",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gemini-2.5-flash",
                    "claude-4-sonnet",
                    "qwen-max",
                    "deepseek-r1",
                    "deepseek-v3",
                    "qwen3-235b-a22b",
                    "qwen3-32b",
                    "qwen3-14b",
                    "qwen3-8b",
                    "qwen3-4b",
                    "gemma-3-27b-it",
                    "gemma-3-12b-it",
                    "gemma-3-4b-it",
                ]
            ):
                rollout_file = f"data/rollout_{task}_{model}.jsonl"
                if task == "web":
                    rollout_file = f"data/simpleqa_rollout_{task}_{model}.jsonl"
                result = rollout_data_insight(rollout_file, test_size)
                task_result[model] = result
            for key in task_result[list(task_result.keys())[0]].keys():
                for model, result in task_result.items():
                    print(f"{model} {key}: {result[key]:.3f}")
                print("-" * 20)
