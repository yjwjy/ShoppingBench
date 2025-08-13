import sys
import ujson as json
from collections import defaultdict

from tqdm import tqdm

from run_evaluate import (
    load_rollout_outputs,
    load_synthesize_rewards,
    load_synthesize_vouchers,
    eval_product,
    eval_shop,
    eval_voucher
)


def case_study(config: dict):
    task = config["task"]
    rewards = load_synthesize_rewards(config)
    vouchers = load_synthesize_vouchers(config)
    outputs = load_rollout_outputs(config)
    if "ablation_react" in config["rollout_file"]:
        mode = "no think"
    else:
        mode = "think"

    samples = []
    for query in tqdm(rewards.keys()):
        if query not in outputs:
            continue

        output = outputs[query]
        reward = rewards[query]
        voucher = vouchers.get(query)

        is_error = False

        score = defaultdict(float)
        if task == "product":
            eval_product(score, output, reward)
            if score["rule"] < 1:
                is_error = True
        elif task == "shop":
            eval_shop(score, output, reward)
            if score["rule"] < 1 or score["shop"] < 1:
                is_error = True
        elif task == "voucher":
            eval_voucher(score, output, reward, voucher)
            if score["rule"] < 1 or score["budget"] < 1:
                is_error = True
        else:
            raise Exception(f"Invalid task: {task}")

        if not is_error:
            continue

        samples.append({
            "task": task,
            "query": query,
            "score": score,
            "rollout": output,
            "reward": reward,
            "voucer": voucher
        })

    with open(config["case_study_file"], "w") as fout:
        for sample in samples:
            fout.write(f"{json.dumps(sample, indent=2)}\n")


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    case_study(config)
