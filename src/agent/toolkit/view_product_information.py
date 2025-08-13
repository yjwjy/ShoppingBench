import time
import logging

import requests

from .base import BaseTool


TIMEOUT = 60

MAX_RETRIES = 10

DESC = """Given a list of product_ids (unique product identifiers), fetch their corresponding information, including product descriptions, SKU options, and SPU attributes."""

PRODUCT_IDS_DESC = """A comma-separated list of product_ids (unique product identifiers)."""

logger = logging.getLogger(__name__)


class ViewProductInformation(BaseTool):
    name: str = "view_product_information"
    description: str = DESC
    parameters: dict = {
        "type": "object",
        "properties": {
            "product_ids": {"type": "string", "description": PRODUCT_IDS_DESC},
        },
        "required": ["product_ids"],
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
            "product_ids": data["product_ids"],
        }

        # url
        url = "http://127.0.0.1:5631/view_product_information?"
        url += "&".join("{}={}".format(str(k), str(v)) for k, v in params.items())

        # request
        return requests.get(url, timeout=TIMEOUT)

    def _parse_response(self, response):
        return response.json()

    def execute(self, **kwargs):
        return self._execute(kwargs)
