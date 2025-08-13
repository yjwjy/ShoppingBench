import sys
import random
import ujson as json
from collections import defaultdict

from tqdm import tqdm

from run_evaluate import (
    load_rollout_outputs,
    load_synthesize_rewards,
    load_synthesize_vouchers,
    eval_product,
    eval_shop,
    eval_voucher,
)
from util.message import Message


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

    samples = []
    for query in tqdm(rewards.keys()):
        all_outputs = []
        max_outputs = []
        for outputs, mode in zip(outputs_list, mode_list):
            if query not in outputs:
                continue

            output = outputs[query]
            reward = rewards[query]
            voucher = vouchers.get(query)

            all_outputs.append(output)

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
                if score["rule"] < 1 and score["budget"] < 1:
                    continue
            else:
                raise Exception(f"Invalid task: {task}")

            max_outputs.append(output)

        if not max_outputs:
            max_output = random.choice(all_outputs)
        else:
            max_output = random.choice(max_outputs)

        corpus_tracker = []
        index = 1
        for step in max_output:
            message = Message.from_dict(step["completion"]["message"])
            if message.tool_call and message.obs:
                for commend, observation in zip(message.tool_call, message.obs):
                    del commend["tool_call_id"]
                    del observation["tool_call_id"]
                    corpus_tracker.append(
                        {
                            "prompt": [
                                {"role": "system", "content": ""},
                                {"role": "user", "content": ""},
                            ],
                            "completion": {
                                "reasoning_content": "",
                                "content": "",
                                "message": {
                                    "tool_call": [commend],
                                    "obs": [observation],
                                },
                            },
                            "extra_info": {
                                "step": index,
                                "query": query,
                                "timestamp": None,
                            },
                        }
                    )
                    index += 1

        samples.append(corpus_tracker)
    with open(config["multi_vote_file"], "w") as fout:
        for sample in samples:
            fout.write(f"{json.dumps(sample)}\n")


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    reject_sample(config)
