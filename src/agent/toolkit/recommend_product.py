from .base import BaseTool


DESC = """"Recommend the products to the user. You can use the tool only once."""

PRODUCT_IDS_DESC = """A comma-separated list of product_ids:
1. If the user finds a single product, provide the product_id that best matches the user's requirements.
2. If the user finds `N` products, provide `N` product_ids in the order specified by the user's requirements.
3. If the user finds a shop selling `N` products, provide `N` product_ids in the specified order, ensuring they all come from the same shop."""


class RecommendProduct(BaseTool):
    name: str = "recommend_product"
    description: str = DESC
    parameters: dict = {
        "type": "object",
        "properties": {
            "product_ids": {"type": "string", "description": PRODUCT_IDS_DESC}
        },
        "required": ["product_ids"],
    }

    def execute(self, **kwargs):
        product_ids = kwargs.get("product_ids", "")
        return f"Having recommended the products to the user: {product_ids}."
