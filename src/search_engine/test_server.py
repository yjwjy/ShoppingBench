import ujson as json

import requests

timeout = 60

url = "http://127.0.0.1:5631/find_product?q=red+nike+shoes&page=1&price=0-5000&sort=order&service=official&shop_id=1706996"
resp = requests.get(url, timeout=timeout)
print(f"find_product:\n{json.dumps(resp.json(), indent=4)}")

url = "http://127.0.0.1:5631/view_product_information?product_ids=3134952883,4321744324"
resp = requests.get(url, timeout=timeout)
print(f"view_product_info:\n{json.dumps(resp.json(), indent=4)}")