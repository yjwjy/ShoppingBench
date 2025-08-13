import sys
import ujson as json
from collections import defaultdict

import tiktoken
from tqdm import tqdm

from run_evaluate import (
    load_rollout_outputs,
    load_synthesize_rewards,
    load_synthesize_vouchers,
    eval_product,
    eval_shop,
    eval_voucher
)
from rewards.orm import length_reward
from rewards.prm import format_reward
from util.message import Message, OUTPUT_ROLES


enc = tiktoken.encoding_for_model("gpt-4o")


def print_data_insight(samples: list[dict]):
    instruction_tokens = []
    input_tokens = []
    output_tokens = []
    all_tokens = []
    for data in samples:
        instruction = data["instruction"]
        input = data["input"]
        output = data["output"]

        instruction_tokens.append(len(enc.encode(instruction)))
        input_tokens.append(len(enc.encode(input)))
        output_tokens.append(len(enc.encode(output)))
        all_tokens.append(len(enc.encode(instruction)) + len(enc.encode(input)) + len(enc.encode(output)))
    instruction_tokens.sort()
    input_tokens.sort()
    output_tokens.sort()
    all_tokens.sort()

    print(f"#Samples: {len(samples)}")
    print(f"Instruction #Tokens Min/Median/Max: {instruction_tokens[0]}/{instruction_tokens[int(0.5 * len(instruction_tokens))]}/{instruction_tokens[-1]}")
    print(f"Input #Tokens Min/Median/Max: {input_tokens[0]}/{input_tokens[int(0.5 * len(input_tokens))]}/{input_tokens[-1]}")
    print(f"Output #Tokens Min/Median/Max: {output_tokens[0]}/{output_tokens[int(0.5 * len(output_tokens))]}/{output_tokens[-1]}")
    print(f"All #Tokens Min/Median/Max: {all_tokens[0]}/{all_tokens[int(0.5 * len(all_tokens))]}/{all_tokens[-1]}")


def reject_sample(config: dict):
    task = config["task"]
    rewards = load_synthesize_rewards(config)
    vouchers = load_synthesize_vouchers(config)
    outputs_list = []
    mode_list = []
    for rollout_file in config["rollout_files"]:
        outputs = load_rollout_outputs({"rollout_file": rollout_file})
        outputs_list.append(outputs)
        if "ablation_react" in rollout_file:
            mode_list.append("no think")
        else:
            mode_list.append("think")

    total = set()
    success_ns = set()
    format_ns = set()
    tokens_ns = set()
    samples = []
    for query in tqdm(rewards.keys()):
        max_output = None
        max_length_score = 0
        max_mode = ""
        for outputs, mode in zip(outputs_list, mode_list):
            if query not in outputs:
                continue
            total.add(query)

            output = outputs[query]
            reward = rewards[query]
            voucher = vouchers.get(query)

            length_score = length_reward(output)
            score = defaultdict(float)
            if task == "product":
                eval_product(score, output, reward)
                if score["rule"] < 1:
                    continue
            elif task == "shop":
                eval_shop(score, output, reward)
                if score["rule"] < 1 or score["shop"] < 1:
                    continue
            elif task == "voucher":
                eval_voucher(score, output, reward, voucher)
                if score["rule"] < 1 or score["budget"] < 1:
                    continue
            else:
                raise Exception(f"Invalid task: {task}")

            if length_score > max_length_score:
                max_output = output
                max_length_score = length_score
                max_mode = mode

        if not max_output:
            continue
        success_ns.add(query)

        for step in max_output:
            message = Message.from_dict(step["completion"]["message"])
            if message.tool_call:
                for commend in message.tool_call:
                    if "tool_call_id" in commend:
                        del commend["tool_call_id"]
            completion = message.to_string(OUTPUT_ROLES)

            if not completion:
                continue

            format_reward_score = format_reward(completion) if max_mode == "think" else format_reward(completion, ["tool_call"])
            if format_reward_score < 1:
                continue
            format_ns.add(query)

            system_prompt = [x["content"] for x in step["prompt"] if x["role"] == "system"][0]
            user_prompt = [x["content"] for x in step["prompt"] if x["role"] == "user"][0]
            data = {
                "instruction": system_prompt,
                "input": user_prompt,
                "output": completion,
            }
 
            instruction_tokens = enc.encode(data["instruction"])
            input_tokens = enc.encode(data["input"])
            output_tokens = enc.encode(data["output"])
            if len(instruction_tokens) + len(input_tokens) + len(output_tokens) > 16 * 1024:
                continue
            tokens_ns.add(query)

            samples.append(data)

    with open(config["rs_file"], "w") as fout:
        fout.write(f"{json.dumps(samples)}\n")

    print(f"Total #Queries: {len(total)}, Success NS #Queries: {len(success_ns)}, Format NS #Queries: {len(format_ns)}, Tokens NS #Queries: {len(tokens_ns)}")
    print_data_insight(samples)


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    reject_sample(config)
