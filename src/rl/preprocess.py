import re
import os
import json
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm

np.random.seed(31415)

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_length(tokenizer, instruction, inputs):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": inputs},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt")

    return model_inputs.input_ids.shape[1]


def filter_length(data, tokenizer, max_length):
    passed_data = []
    for item in tqdm(data):
        input_length = get_length(tokenizer, item["instruction"], item["input"])
        if input_length < max_length:
            passed_data.append(item)
    return passed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_length", default=16384)
    parser.add_argument("--val_size", default=0.1)
    args = parser.parse_args()

    # Load dataset
    dataset = json.load(open(args.dataset_file, "r"))
    train_num = int(len(dataset) * (1 - args.val_size))
    dataset_train = dataset[:train_num]
    dataset_test = dataset[train_num:]

    # Shuffle dataset
    np.random.shuffle(dataset_train)

    # Filter dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = filter_length(dataset_train, tokenizer, args.max_length)
    test_dataset = filter_length(dataset_test, tokenizer, args.max_length)

    # Function to process each example
    def process_fn(example, idx, split):
        instruction = example["instruction"]
        input_text = example["input"]
        output = example["output"]

        data = {
            "data_source": "shoppingbench",
            "prompt": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ],
            "ability": "shopping",
            "reward_model": {"style": "rule", "ground_truth": output},
            "extra_info": {
                "split": split,
                "index": idx,
                "instruction": instruction,
                "input": input_text,
                "output": output,
            },
        }
        return data

    # Process dataset using list comprehension
    train_dataset = [process_fn(d, idx, "train") for idx, d in enumerate(train_dataset)]
    test_dataset = [process_fn(d, idx, "test") for idx, d in enumerate(test_dataset)]

    # Convert to Pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    # Save as Parquet
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"Saved datasets to {local_dir}")
