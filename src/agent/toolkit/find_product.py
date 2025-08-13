import time
import logging
from urllib.parse import quote_plus

import requests

from .base import BaseTool


TIMEOUT = 60

MAX_RETRIES = 10

DESC = """Search for products and return up to 10 products, with each product including a product_id, shop_id, title, price, service, and sold_count."""

Q_DESC = """The query used to search for products. e.g. "nike shoes", "backpack for college student"."""

PAGE_DESC = """Modify the parameter, ranging from 1 to 5, to get additional products."""

SHOP_ID_DESC = """Specify a shop using the shop_id, then search for products within that shop."""

PRICE_DESC = """The price range for the products. e.g. "0-100", "100-1000", "1000-"."""

SORT_DESC = """Choose one from the options listed below:
- priceasc: Search for the products and sort by price in ascending order.
- pricedesc: Search for the products and sort by price in descending order.
- order: Search for the products and sort by sales volume in descending order.
- default: Search for the products and sort by the relevance between query and product."""

SERVICE_DESC = """Choose one or more from the options listed below and join them with a comma (","):
- official: Search for products and only select those that are offered with LazMall service. LazMall offers a 100% authenticity guarantee, 15-day unconditional returns, 7-day delivery, and other services.
- freeShipping: Search for products and only select those that are offered with free shipping service.
- COD: Search for products and only select those that are offered with cash on delivery service.
- flashsale: Search for products and only select those that are offered with LazFlash service. LazFlash offers products with limited-time promotions, and its discounts are often significant.
- default: Search for products without applying any other selection criteria."""

logger = logging.getLogger(__name__)


class FindProduct(BaseTool):
    name: str = "find_product"
    description: str = DESC
    parameters: dict = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": Q_DESC},
            "page": {"type": "integer", "description": PAGE_DESC},
            "shop_id": {"type": "string", "description": SHOP_ID_DESC},
            "price": {"type": "string", "description": PRICE_DESC},
            "sort": {
                "type": "string",
                "description": SORT_DESC,
                "enum": ["priceasc", "pricedesc", "order", "default"],
            },
            "service": {"type": "string", "description": SERVICE_DESC},
        },
        "required": ["q", "page"],
    }

    def _execute(self, data):
        for i in range(MAX_RETRIES):
            try:
                resp = self._request(data)
                if resp.status_code == 200:
                    results = self._parse_response(resp)
                    return results
            except Exception as e:
                logger.error(
                    f"Parse lazada search response error, retry {i+1}/{MAX_RETRIES}"
                )
                logger.error("Exception: {}".format(e))
                time.sleep(3)

    def _request(self, data):
        params = {
            "q": data["q"],
            "page": data["page"],
            "shop_id": data.get("shop_id"),
            "price": data.get("price"),
            "sort": data.get("sort"),
            "service": data.get("service"),
        }

        # preprocess
        params["q"] = quote_plus(params["q"])

        if not params["shop_id"]:
            params.pop("shop_id")

        if not params["price"]:
            params.pop("price")

        if not params["sort"] or "default" == params["sort"]:
            params.pop("sort")

        if not params["service"] or "default" == params["service"]:
            params.pop("service")
        elif "default" in params["service"]:
            params["service"] = ",".join(
                x for x in params["service"].split(",") if x != "default"
            )

        # url
        url = "http://127.0.0.1:5631/find_product?"
        url += "&".join("{}={}".format(str(k), str(v)) for k, v in params.items())

        # request
        return requests.get(url, timeout=TIMEOUT)

    def _parse_response(self, response):
        return response.json()

    def execute(self, **kwargs):
        return self._execute(kwargs)
