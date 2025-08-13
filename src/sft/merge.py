import sys
import re
import random
import ujson as json

import tiktoken
from tqdm import tqdm

enc = tiktoken.encoding_for_model("gpt-4o")

input_files = sys.argv[1]
output_file = sys.argv[2]

samples = []
for input_file in input_files.split(","):
    queries = set()
    think_count = 0
    no_think_count = 0
    input_tokens = 0
    output_tokens = 0
    with open(input_file, "r") as fin:
        jsonobj = json.loads(fin.read())
        for sample in tqdm(jsonobj):
            instruction = sample["instruction"]
            input_ = sample["input"]
            output = sample["output"]
            query = re.search("<user>(.+)</user>", input_, re.DOTALL).group(1).strip()
            queries.add(query)
            think_count += 1 if "1. Your output must always include `<think>...</think>` and at least one of `<tool_call>...</tool_call>` or `<response>...</response>`. No other content is allowed." in instruction else 0
            no_think_count += 1 if "only include `<tool_call>...</tool_call>` and nothing else." in instruction else 0
            input_tokens += len(enc.encode(instruction)) + len(enc.encode(input_))
            output_tokens += len(enc.encode(output))
            samples.append(sample)
    print(f"{input_file}: #queries={len(queries)}, #think={think_count}, #no think={no_think_count}, #input tokens={input_tokens}, #output tokens={output_tokens}")

with open(output_file, "w") as fout:
    random.shuffle(samples)
    json.dump(samples, fout)
