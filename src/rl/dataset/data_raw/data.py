# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the shoppingbench dataset to parquet format
"""

import re
import os
import json
import numpy as np
import pandas as pd
import argparse

np.random.seed(31415)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)




from tqdm import tqdm

def get_length(tokenizer, instruction, inputs):
    messages = [
        {
            "role": "system",
            "content": instruction
        },
        {
            "role": "user",
            "content": inputs
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt")

    return model_inputs.input_ids.shape[1]

def filter_length(data, tokenzier):
    passed_data = []
    for item in tqdm(data):
        input_length = get_length(tokenizer, item['instruction'], item['input'])
        if input_length < 8192:
            passed_data.append(item)
    return passed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./dataset/shoppingbench')
    args = parser.parse_args()
    
    data_source = 'shoppingbench'

    # Load dataset
    dataset_train = json.load(open("./dataset/data_raw/rs2_merge_train_psvk.json", "r"))
    dataset_test = json.load(open("./dataset/data_raw/rs2_merge_test_psvk.json", "r"))

    # Shuffle dataset
    np.random.shuffle(dataset_train)

    train_dataset = filter_length(dataset_train, tokenizer)
    test_dataset = filter_length(dataset_test, tokenizer)

    # Function to process each example
    def process_fn(example, idx, split):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]

        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": output
            },
            "extra_info": {
                'split': split,
                'index': idx,
                "instruction": instruction,
                "input": input_text,
                "output": output,
            }
        }
        return data

    # Process dataset using list comprehension
    train_dataset = [process_fn(d, idx, 'train') for idx, d in enumerate(train_dataset)]
    test_dataset = [process_fn(d, idx, 'test') for idx, d in enumerate(test_dataset)]

    # Convert to Pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    # Save as Parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}")