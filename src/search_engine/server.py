import os
import sys
import ujson as json
import multiprocessing

from pyserini.search.lucene import LuceneSearcher
from flask import Flask, request, jsonify
from waitress import serve

searcher = LuceneSearcher("indexes")
print("Load indexes done.", file=sys.stderr)

app = Flask(__name__)


CAPACITY = 100000
PAGE_SIZE = 10
MAX_PAGE = 5
SEARCH_FIELDS = ["product_id", "shop_id", "title", "price", "service", "sold_count"]
INFORMATION_FIELDS = [
    "product_id",
    "short_description",
    "description",
    "sku_options",
    "attributes",
]


def convert_str_to_float(x):
    try:
        x = float(x)
    except:
        x = None
    return x


def process_page(page):
    if not page or not page.isdigit() or not (1 <= int(page) <= MAX_PAGE):
        return None
    return int(page)


def process_price(price):
    if not price or "-" not in price:
        return [None, None]

    splited = price.split("-")
    if len(splited) != 2:
        return [None, None]

    low, high = splited
    low = convert_str_to_float(low)
    high = convert_str_to_float(high)
    return [low, high]


def process_sort(sort):
    if sort not in {"order", "priceasc", "pricedesc"}:
        return None
    return sort


def process_service(service):
    results = []
    if not service:
        return results
    for serv in service.split(","):
        if serv not in {"official", "freeShipping", "COD", "flashsale"}:
            continue
        if serv in results:
            continue
        results.append(serv)
    return results


def is_filter_by_price(product, price):
    low, high = price
    if low is not None and product["price"] < low:
        return True
    if high is not None and product["price"] > high:
        return True
    return False


def is_filter_by_service(product, service):
    for serv in service:
        if serv not in product.get("service", []):
            return True
    return False


def is_filter_by_shop_id(product, shop_id):
    if shop_id and shop_id != product.get("shop_id"):
        return True
    return False


def search(q, page, shop_id=None, price=None, sort=None, service=None):
    page = process_page(page)
    price = process_price(price)
    sort = process_sort(sort)
    service = process_service(service)

    products = []

    # page
    if page is None:
        return products

    # filter by shop_id & price & service
    hits = searcher.search(q=q, k=CAPACITY, remove_dups=True)
    for hit in hits:
        product = json.loads(searcher.doc(hit.docid).raw())["product"]
        if is_filter_by_shop_id(product, shop_id):
            continue
        if is_filter_by_price(product, price):
            continue
        if is_filter_by_service(product, service):
            continue
        products.append(product)
        if len(products) >= MAX_PAGE * PAGE_SIZE:
            break

    # sort
    if sort == "order":
        products.sort(key=lambda x: x["sold_count"], reverse=True)
    elif sort == "priceasc":
        products.sort(key=lambda x: x["price"], reverse=False)
    elif sort == "pricedesc":
        products.sort(key=lambda x: x["price"], reverse=True)

    a_page = products[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
    results = []
    for product in a_page:
        results.append({k: product[k] for k in SEARCH_FIELDS})
    return results


def information(product_ids, delimiter=","):
    results = []

    product_ids = product_ids.split(delimiter)
    if len(product_ids) == 0:
        return results

    for product_id in product_ids:
        doc = searcher.doc(product_id)
        if not doc:
            continue
        product = json.loads(doc.raw())["product"]
        results.append({k: product[k] for k in INFORMATION_FIELDS})
    return results


@app.route("/")
def index():
    usage = {
        "/find_product": "q,page,shop_id,price,sort,service",
        "/view_product_information": "product_ids",
    }
    return jsonify(usage)


@app.route("/find_product")
def find_product():
    result = search(
        q=request.args.get("q"),
        page=request.args.get("page"),
        shop_id=request.args.get("shop_id"),
        price=request.args.get("price"),
        sort=request.args.get("sort"),
        service=request.args.get("service"),
    )
    return jsonify(result)


@app.route("/view_product_information")
def view_product_information():
    result = information(product_ids=request.args.get("product_ids"))
    return jsonify(result)


if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    threads = max(4, cores)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5631"))

    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        expose_tracebacks=True,
        channel_timeout=60,
        cleanup_interval=10,
    )
