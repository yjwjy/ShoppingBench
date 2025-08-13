import sys
import ujson as json

input_file = sys.argv[1]

with open(input_file, "r") as fin:
    jsonobj = json.load(fin)
    for sample in jsonobj:
        instruction = sample["instruction"]
        input_ = sample["input"]
        output = sample["output"]

        print(f"====== instruction ======\n{instruction}\n====== input ======\n{input_}\n====== output ======\n{output}")

