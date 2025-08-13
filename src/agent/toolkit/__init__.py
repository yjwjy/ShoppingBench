from toolkit.find_product import FindProduct
from toolkit.view_product_information import ViewProductInformation
from toolkit.recommend_product import RecommendProduct
from toolkit.terminate import Terminate
from toolkit.python_execute import PythonExecute
from toolkit.web_search import WebSearch

toolkit = [
    FindProduct,
    ViewProductInformation,
    RecommendProduct,
    Terminate,
    WebSearch,
]
tools = [toolclass() for toolclass in toolkit]
toolmap = {tool.name: tool for tool in tools}
