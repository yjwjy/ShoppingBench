import os
import sys
import html
import copy
import ujson as json
from collections import defaultdict, Counter

from tqdm import tqdm


products_file = sys.argv[1]
documents_file = sys.argv[2]
total_num_lines = int(os.popen(f"wc -l {products_file}").read().strip().split(" ", 1)[0])

attr_k_counter = Counter()
attr_v_counter = Counter()


def is_contain_rubbish_words(text: str):
    rubbish_words = [
        "NoBrand",
        "No Brand",
        "NotBrand",
        "Not Brand",
        "NotBranded",
        "Not Branded",
        "NA",
        "N/A",
        "NULL",
        "No",
        "NoSpecified",
        "No Specified",
        "NotSpecified",
        "Not Specified",
        "Other",
        "Others",
        "Unknown",
        "Miscellaneous",
        "New",
    ]
    for word in rubbish_words:
        if word.lower() == text.lower():
            return True
    return False


def process_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text).strip()
    return text[:2000]


def process_brand(brand_name: str) -> str:
    if brand_name and not is_contain_rubbish_words(brand_name):
        return brand_name
    return ""


def process_sku(sku: str) -> dict:
    if not sku:
        return dict()

    result = defaultdict(dict)
    for id_k_v in sku.split(chr(2)):
        splited = id_k_v.split(chr(1))
        if len(splited) != 3:
            continue
        sku_id, sku_k, sku_v = (x.lower() for x in splited)
        if is_contain_rubbish_words(sku_v):
            continue
        result[sku_id][sku_k] = sku_v

    index = 1
    new_result = dict()
    for sku_id, kv in result.items():
        if index >= 10:
            break

        new_result[index] = kv
        index += 1
    return new_result


def process_spu(spu: str) -> dict:
    if not spu:
        return dict()

    result = defaultdict(set)
    for k_v in spu.split(chr(2)):
        splited = k_v.split(chr(1))
        if len(splited) != 2:
            continue
        spu_k, spu_v = (x.lower() for x in splited)
        if is_contain_rubbish_words(spu_v):
            continue
        result[spu_k].add(spu_v)
    return {k: list(v) for k, v in result.items()}


def process_cpv(cpv: str) -> dict:
    if not cpv:
        return dict()

    result = dict()
    for name_values_isImportant_isKeyAttribute in cpv.split(chr(3)):
        splited = name_values_isImportant_isKeyAttribute.split(chr(2))
        if len(splited) != 4:
            continue
        name, values, is_important, is_key_attribute = (x.lower() for x in splited)
        if is_important != "y" or is_key_attribute != "y":
            continue
        result[name] = []
        for value in values.split(chr(1)):
            if is_contain_rubbish_words(value):
                continue
            result[name].append(value)
    return result


def merge_attributes(processed_spu: dict, processed_cpv: dict) -> dict:
    if not processed_spu and not processed_cpv:
        return dict()
    elif not processed_spu:
        return processed_cpv
    elif not processed_cpv:
        return processed_spu

    result = copy.deepcopy(processed_cpv)
    for k, vs in processed_spu.items():
        if k not in result:
            result[k] = vs
        else:
            for v in vs:
                if v not in result[k]:
                    result[k].append(v)
    return result


def process_products():
    with open(products_file, "r") as fin, open(f"{products_file}.processed", "w") as fout:
        for line in tqdm(fin, total=total_num_lines):
            p = json.loads(line.strip())
            product = dict()

            # id
            product["product_id"] = p["product_id"]
            product["shop_id"] = p["shop_id"]

            # brand
            product["brand"] = process_brand(p["brand_name"])

            # category
            product["category"] = p["category_hierarchy"].replace("-", " > ")

            # text
            product["title"] = process_text(p["title"])
            product["short_description"] = process_text(p["short_description"])
            product["description"] = process_text(p["description"])
            product["specification"] = process_text(p["specification"])

            # price
            product["price"] = float(p["price"])

            # sold count
            product["sold_count"] = int(p["sold_cnt"])

            # Stock Keeping Unit Options
            sku_options = process_sku(p["sku"])
            product["sku_options"] = sku_options
            for _, kv in sku_options.items():
                for k, v in kv.items():
                    attr_k_counter[k] += 1
                    attr_v_counter[v] += 1

            # attributes
            processed_spu = process_spu(p["spu"])
            processed_cpv = process_cpv(p["cpv"])
            attributes = merge_attributes(processed_spu, processed_cpv)
            product["attributes"] = attributes
            for k, vs in attributes.items():
                attr_k_counter[k] += 1
                for v in vs:
                    attr_v_counter[v] += 1

            # main image url
            product["main_image_url"] = p["main_image_url"]

            # product url
            product["product_url"] = p["product_url"]

            # service
            product["service"] = p["service"]

            fout.write(json.dumps(product) + "\n")


def filter_longtail_attrs():
    threshold = 20
    with open(f"{products_file}.processed", "r") as fin, open(f"{products_file}.processed.tmp", "w") as fout:
        for line in tqdm(fin, total=total_num_lines):
            product = json.loads(line.strip())

            sku_options = product["sku_options"]
            new_sku_options = dict()
            index = 1
            for _, kv in sku_options.items():
                new_kv = dict()
                for k, v in kv.items():
                    if attr_k_counter[k] >= threshold and attr_v_counter[v] >= threshold:
                        new_kv[k] = v
                if new_kv:
                    new_sku_options[index] = new_kv
                    index += 1
            product["sku_options"] = new_sku_options
 
            attributes = product["attributes"]
            new_attributes = dict()
            for k, vs in attributes.items():
                if attr_k_counter[k] < threshold:
                    continue
                new_vs = []
                for v in vs:
                    if attr_v_counter[v] < threshold:
                        continue
                    new_vs.append(v)
                if new_vs:
                    new_attributes[k] = new_vs
            product["attributes"] = new_attributes

            fout.write(f"{json.dumps(product)}\n")
    os.system(f"mv {products_file}.processed.tmp {products_file}.processed")


def write_documents():
    with open(f"{products_file}.processed", "r") as fin, open(f"{documents_file}.tmp", "w") as fout:
        for line in tqdm(fin, total=total_num_lines):
            p = json.loads(line.strip())

            contents = []
            contents.append(p["title"])

            sku_values = set()
            for kv in p["sku_options"].values():
                for v in kv.values():
                    sku_values.add(v)
            contents.append(" ".join(sku_values))

            attributes = set()
            for vs in p["attributes"].values():
                for v in vs:
                    attributes.add(v)
            contents.append(" ".join(attributes))

            doc = {"id": p["product_id"], "contents": "\n".join(contents), "product": p}
            fout.write(json.dumps(doc) + "\n")

    os.system(f"mv {documents_file}.tmp {documents_file}")
    os.system(f"rm -rf {products_file}.processed")


if __name__ == "__main__":
    process_products()
    filter_longtail_attrs()
    write_documents()
