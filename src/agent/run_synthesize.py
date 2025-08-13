import os
import re
import sys
import math
import random
import ujson as json
from collections import defaultdict

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

from util.llm import ask_llm


random.seed(42)
searcher = LuceneSearcher("indexes")


def load_sid2pids(config: dict) -> dict:
    shop2products = defaultdict(list)
    total = int(
        os.popen(f"wc -l {config['documents_file']}").read().strip().split(" ", 1)[0]
    )
    with open(config["documents_file"], "r") as fin:
        for line in tqdm(fin, total=total, desc="Load `shop -> products` mappings: "):
            jsonobj = json.loads(line.strip())
            product = jsonobj["product"]
            shop_id = product["shop_id"]
            product_id = product["product_id"]
            shop2products[shop_id].append(product_id)
    return shop2products


def load_pids(config: dict) -> list:
    products = []
    total = int(
        os.popen(f"wc -l {config['documents_file']}").read().strip().split(" ", 1)[0]
    )
    with open(config["documents_file"], "r") as fin:
        for line in tqdm(fin, total=total, desc="Load products: "):
            jsonobj = json.loads(line.strip())
            product = jsonobj["product"]
            product_id = product["product_id"]
            products.append(product_id)
    return products


def sample_products_in_shop(shop2products: dict, N: int) -> tuple:
    shop_ids = list(shop2products.keys())
    for _ in range(100):
        shop_id = random.choice(shop_ids)

        product_ids = shop2products[shop_id]
        if len(product_ids) < N:
            continue

        selected_product_ids = random.sample(product_ids, N)
        if len(selected_product_ids) != N:
            continue

        products = [
            json.loads(searcher.doc(x).raw())["product"]
            for x in selected_product_ids
            if searcher.doc(x)
        ]
        if len(products) != N:
            continue

        return shop_id, selected_product_ids, products
    return None, None, None


def sample_title(index: int, title: str):
    objs = []
    texts = []
    objs.append(title)
    texts.append(f"{index}. The `title` is `{title}`.")
    index += 1
    return index, objs, texts


def sample_sku_options(index: int, sku_options: dict, multiplier: int):
    assert isinstance(multiplier, int) and multiplier >= 1, "multiplier must be a positive integer"
    objs = []
    texts = []
    for option in sku_options.values():
        if random.randint(0, multiplier) == 1:
            objs.append(option)
            for key, value in option.items():
                texts.append(f"{index}. The `{key}` is `{value}`.")
                index += 1
            break
    return index, objs, texts


def sample_attributes(index: int, attributes: dict, multiplier: int):
    assert isinstance(multiplier, int) and multiplier >= 1, "multiplier must be a positive integer"
    objs = []
    texts = []
    for key, values in attributes.items():
        if not key or not values:
            continue
        if random.randint(0, multiplier) == 1:
            selected_values = random.sample(values, 1)
            objs.append({key: selected_values})
            texts.append(f"{index}. The `{key}` is `{', '.join(selected_values)}`")
            index += 1
    return index, objs, texts


def sample_service(index: int, service: list, multiplier: int):
    assert isinstance(multiplier, int) and multiplier >= 1, "multiplier must be a positive integer"
    service_mapping = {
        "official": "",
        "freeShipping": "The product offer with free shipping service.",
        "COD": "The product offer with cash on delivery service.",
        "flashsale": "",
    }

    objs = []
    texts = []
    for serv in service:
        if random.randint(0, multiplier) == 1:
            objs.append(serv)
            texts.append(f"{index}. {service_mapping[serv]}")
            index += 1
    return index, objs, texts


def sample_price(index: int, price: float, multiplier: int):
    assert isinstance(multiplier, int) and multiplier >= 1, "multiplier must be a positive integer"
    modes = ["less than", "greater than", "between"]
    objs = []
    texts = []
    for mode in modes:
        if random.randint(0, multiplier) == 1:
            if mode == "less than":
                lower_bound = 0
                upper_bound = int(price + random.randint(0, int(price)))
                text = f"{mode} `{upper_bound}` Philippine Peso"
            elif mode == "greater than":
                lower_bound = int(price - random.randint(0, int(price)))
                upper_bound = None
                text = f"{mode} `{lower_bound}` Philippine Peso"
            else:
                lower_bound = int(price - random.randint(0, int(price)))
                upper_bound = int(price + random.randint(0, int(price)))
                text = f"{mode} `{lower_bound}` and `{upper_bound}` Philippine Peso"

            if lower_bound < 0 or price <= lower_bound:
                continue
            if upper_bound is not None and price >= upper_bound:
                continue

            objs.append({mode: [lower_bound, upper_bound]})
            texts.append(f"{index}. The `price` {text}.")
            index += 1
            break
    return index, objs, texts


def generate_target_product(
    product: dict, multiplier: int, exculde: list = []
) -> tuple:
    reward = dict()
    requirement = []
    index = 1

    # product_id
    reward["product_id"] = product["product_id"]

    # title
    if "title" not in exculde:
        title = product["title"]
        if title:
            index, objs, texts = sample_title(index, title)
            if objs:
                reward["title"] = objs
            if texts:
                requirement.extend(texts)

    # sku options
    if "sku_options" not in exculde:
        sku_options = product["sku_options"]
        if len(sku_options) > 0:
            index, objs, texts = sample_sku_options(index, sku_options, multiplier)
            if objs:
                reward["sku_options"] = objs
            if texts:
                requirement.extend(texts)

    # spu attributes
    if "attributes" not in exculde:
        attributes = product["attributes"]
        if len(attributes) > 0:
            index, objs, texts = sample_attributes(index, attributes, multiplier)
            if objs:
                reward["attributes"] = objs
            if texts:
                requirement.extend(texts)

    # service
    if "service" not in exculde:
        service = product["service"]
        if len(service) > 0:
            index, objs, texts = sample_service(index, service, multiplier)
            if objs:
                reward["service"] = objs
            if texts:
                requirement.extend(texts)

    # price
    if "price" not in exculde:
        price = product["price"]
        if price > 0:
            index, objs, texts = sample_price(index, price, multiplier)
            if objs:
                reward["price"] = objs
            if texts:
                requirement.extend(texts)

    return reward, requirement


def generate_query_and_write(prompt, reward, fout, external="", voucher=None) -> bool:
    reasoning_content, content = ask_llm(
        messages=[{"role": "user", "content": prompt}],
        model_config=config["model_config"],
    )

    query = ""
    matchobj = re.search("```json(.+?)```", content, re.DOTALL)
    if matchobj:
        jsonstr = matchobj.group(1).strip()
        jsonobj = json.loads(jsonstr)
        query = jsonobj.get("query")
        knowledge_point = jsonobj.get("knowledge point")

    if not query:
        return False

    if external:
        query = f"{query}\n\n{external}"

    # write
    data = {"prompt": prompt, "query": query, "reward": reward}
    if voucher:
        data["voucher"] = voucher
    jsonstr = json.dumps(data)
    fout.write(f"{jsonstr}\n")
    fout.flush()
    return True


def synthesize_product(config: dict):
    with open(config["synthesize_prompt_file"], "r") as fin:
        prompt_template = fin.read().strip()

    pids = load_pids(config)

    total = config["total"]
    count = 0
    used = set()
    pbar = tqdm(total=total, desc="Generate product intention queries: ")
    with open(config["synthesize_file"], "w") as fout:
        while count < total:
            # 1. Product selection
            product_id = random.choice(pids)
            if product_id in used:
                continue

            doc = searcher.doc(product_id)
            if not doc:
                continue
            jsonobj = json.loads(doc.raw())
            product = jsonobj["product"]

            # 2. Fields selection
            reward, requirement = generate_target_product(product, multiplier=1)
            if not requirement:
                continue

            # 3. Query generation
            prompt = prompt_template \
                .replace("<|task|>", "a product") \
                .replace("<|requirements|>", "\n".join(requirement))

            if not generate_query_and_write(prompt, reward, fout):
                continue

            count += 1
            used.add(product_id)
            pbar.update(1)


def synthesize_shop(config: dict):
    with open(config["synthesize_prompt_file"], "r") as fin:
        prompt_template = fin.read().strip()

    sid2pids = load_sid2pids(config)

    total = config["total"]
    count = 0
    used = set()
    pbar = tqdm(
        total=total,
        desc="Generate shop intention queries: ",
    )
    with open(config["synthesize_file"], "w") as fout:
        while count < total:
            # 1. Products selection
            N = random.randint(2, 4)
            shop_id, selected_product_ids, products = sample_products_in_shop(sid2pids, N)
            if any(not x for x in [shop_id, selected_product_ids, products]):
                continue
            if any(x in used for x in selected_product_ids):
                continue

            # 2. Fields selection
            reward_list = []
            requirement_list = []
            for product in products:
                reward, requirement = generate_target_product(product, multiplier=1)
                reward_list.append(reward)
                requirement_list.append(requirement)

            # 3. Query generation
            requirement_str_list = []
            for i, requirement in enumerate(requirement_list):
                requirement_str = "\n".join(requirement)
                requirement_str_list.append(f"## Product {i+1}\n{requirement_str}")
            prompt = prompt_template \
                .replace("<|task|>", "a shop that sells multiple products") \
                .replace("<|requirements|>", "\n\n".join(requirement_str_list))

            if not generate_query_and_write(prompt, reward_list, fout):
                continue

            count += 1
            used.update(selected_product_ids)
            pbar.update(1)


def synthesize_voucher(config: dict):
    with open(config["synthesize_prompt_file"], "r") as fin:
        prompt_template = fin.read().strip()

    sid2pids = load_sid2pids(config)
    pids = load_pids(config)

    total = config["total"]
    count = 0
    used = set()
    pbar = tqdm(total=total, desc="Generate voucher intention queries: ")
    with open(config["synthesize_file"], "w") as fout:
        while count < total:
            # 1. Voucher type selection
            voucher_type = random.choice(["platform", "shop"])

            # 2. Products selection
            selected_product_ids = []
            products = []
            if voucher_type == "platform":
                N = random.randint(1, 4)
                selected_product_ids = random.sample(pids, N)
                products = [
                    json.loads(searcher.doc(x).raw())["product"]
                    for x in selected_product_ids
                    if searcher.doc(x)
                ]
                if len(products) != N:
                    continue
            elif voucher_type == "shop":
                N = random.randint(2, 4)
                shop_id, selected_product_ids, products = sample_products_in_shop(sid2pids, N)
                if any(not x for x in [shop_id, selected_product_ids, products]):
                    continue
            else:
                raise Exception(f"Unknown voucher type: {voucher_type}")
            if any(x in used for x in selected_product_ids):
                continue

            # 3. Voucher Generation
            ## voucher type
            voucher_description = (
                "1. The voucher only applies to the products from the same shop."
                if voucher_type == "shop"
                else "1. The voucher applies to all products."
            )

            ## threshold
            total_price = sum(product["price"] for product in products)
            if total_price < 100:
                continue
            threshold = random.randint(int(total_price * 0.1), int(total_price * 0.9))
            if threshold > total_price:
                continue
            voucher_description += f"\n2. It is valid only when the total price of the products exceeds `{threshold}`."

            ## discount type
            discount_type = random.choice(["fixed", "percentage"])
            if discount_type == "fixed":
                face_value = random.randint(int(threshold * 0.1), int(threshold * 0.5))
                if face_value > threshold:
                    continue
                price_after_voucher = total_price - face_value
                voucher_description += (f"\n3. It provides a fixed discount of `{face_value}`.")
            elif discount_type == "percentage":
                discount = random.randint(10, 50)
                cap = random.randint(int(threshold * discount / 100.0), threshold)
                if cap > threshold or cap <= threshold * discount / 100.0:
                    continue
                price_after_voucher = max(total_price * (1 - discount / 100.0), total_price - cap)
                voucher_description += f"\n3. It provides a percentage discount of `{discount}%` with a cap of `{cap}`."
            else:
                raise Exception(f"Unknown discount type: {discount_type}")

            ## budget
            budget = math.ceil(price_after_voucher) + random.randint(0, math.ceil(price_after_voucher * 0.1))
            if budget < price_after_voucher:
                continue

            ## voucher
            voucher = {
                "voucher_type": voucher_type,
                "threshold": threshold,
                "discount_type": discount_type,
                "face_value": face_value if discount_type == "fixed" else None,
                "discount": discount / 100.0 if discount_type == "percentage" else None,
                "cap": cap if discount_type == "percentage" else None,
                "price_after_voucher": price_after_voucher,
                "budget": budget,
            }

            # 4. Fields selection
            reward_list = []
            requirement_list = []
            for product in products:
                reward, requirement = generate_target_product(
                    product, multiplier=1, exculde=["price"]
                )
                reward_list.append(reward)
                requirement_list.append(requirement)

            # 5. Query generation
            requirement_str_list = []
            if len(requirement_list) > 1:
                for i, requirement in enumerate(requirement_list):
                    requirement_str = "\n".join(requirement)
                    requirement_str_list.append(f"## Product {i+1}\n{requirement_str}")
            else:
                requirement_str = "\n".join(requirement_list[0])
                requirement_str_list.append(requirement_str)
            prompt = prompt_template \
                .replace("<|task|>", "one or more products") \
                .replace("<|requirements|>", "\n\n".join(requirement_str_list))

            external = f"My budget is only `{budget}`, but I have a voucher with the following rules:\n{voucher_description}"
            if not generate_query_and_write(prompt, reward_list, fout, external=external, voucher=voucher):
                continue

            count += 1
            used.update(selected_product_ids)
            pbar.update(1)


def synthesize_web(config: dict):
    with open(config["synthesize_prompt_file"], "r") as fin:
        prompt_template = fin.read().strip()
    
    total = int(os.popen(f"wc -l {config['documents_file']}").read().strip().split(" ", 1)[0])
    with open(config["documents_file"], "r") as fin, open(config["synthesize_file"], "w") as fout:
        lines = fin.readlines()
        for line in tqdm(lines[:100]):
            jsonobj = json.loads(line.strip())
            product = jsonobj["product"]
            reward, _ = generate_target_product(product)
            index = 1
            requirement = []
            # title
            title = product["title"]
            if title:
                index, objs, texts = sample_title(index, title)
                if texts:
                    requirement.extend(texts)
            # spu attributes
            attributes = product["attributes"]
            if len(attributes) > 0:
                index, objs, texts = sample_attributes(index, attributes)
                if texts:
                    requirement.extend(texts)

            # prompt
            if not requirement:
                continue
            prompt = prompt_template.replace(
                "<|requirements|>", "\n".join(requirement)
            )

            generate_query_and_write(prompt, reward, fout)
    


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)

    task_mapping = {
        "product": synthesize_product,
        "shop": synthesize_shop,
        "voucher": synthesize_voucher,
        "web": synthesize_web,
    }
    task_mapping[config["task"]](config)
