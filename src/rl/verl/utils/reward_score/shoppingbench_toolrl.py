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

import re
import json
import os
from collections import Counter


def format_reward(completion: str, roles: list = []) -> float:
    if not roles:
        roles = ["think", "tool_call", "response"]

    pos = dict()
    for role in roles:
        start = [m.start() for m in re.finditer(f"<{role}>", completion)]
        end = [m.start() for m in re.finditer(f"</{role}>", completion)]
        if start or end:
            pos[role] = (start, end)

    if "think" in roles and "think" not in pos:
        return 0
    if "tool_call" not in pos and "response" not in pos:
        return 0

    for role, (start, end) in pos.items():
        if len(start) != len(end):
            return 0
        if len(start) != 1:
            return 0
        if start[0] >= end[0]:
            return 0

    if "tool_call" in pos:
        try:
            tool_call_str = (
                completion[pos["tool_call"][0][0] : pos["tool_call"][1][0]]
                .replace("<tool_call>", "")
                .replace("</tool_call>", "")
            )
            tool_call = json.loads(tool_call_str)
            if not isinstance(tool_call, list):
                return 0
            for commend in tool_call:
                if "name" not in commend or "parameters" not in commend:
                    return 0
                if not isinstance(commend["name"], str):
                    return 0
                if not isinstance(commend["parameters"], dict):
                    return 0
        except:
            return 0

    for i in range(len(roles)):
        for j in range(len(roles)):
            if i == j:
                continue
            if roles[i] in pos and roles[j] in pos:
                if pos[roles[i]][0][0] < pos[roles[j]][0][0] < pos[roles[i]][1][0]:
                    return 0
                if pos[roles[i]][0][0] < pos[roles[j]][1][0] < pos[roles[i]][1][0]:
                    return 0
    return 1


def check_role(role, text):
    if f"<{role}>" in text and f"</{role}>" in text:
        return True
    return False


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order."""
    if list1 == list2:
        return 1.0

    if os.getenv("REFINEDREWARD", 0) == "1":
        print("REFINEDREWARD is set to 1, so strict match is used")
        if list1 != list2:
            return 0.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


# custoimzed reward functions: format
def customize_format_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, **kwargs
):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step >= 30:
            max_possible_reward = max_possible_reward / 2
            min_possible_reward = min_possible_reward / 2
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward

    # schedule reward
    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = 2 - (2 - max_possible_reward) * step / 150
        min_possible_reward = -2 + (2 + min_possible_reward) * step / 150
        if max_possible_reward < 1.0:
            max_possible_reward = 1.0
        if min_possible_reward > -1.0:
            min_possible_reward = -1.0

    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    # print("\n======= Answer ======= ")
    # print(answer[0])
    # print("\n======= Responses ======= ")
    # for idx, response in enumerate(responses):
    #     print(f"*** Response {idx+1}***\n{response}")

    for response, ans in zip(responses, answer):
        reward = min_possible_reward
        if (
            check_role("tool_call", ans)
            and not check_role("think", ans)
            and not check_role("response", ans)
        ):
            if format_reward(response, ["tool_call"]) >= 1:
                reward = max_possible_reward
        else:
            if format_reward(response) >= 1:
                reward = max_possible_reward

        rewards.append(reward)

    # print("\n======= Reward for <format> =======")
    # print("Reward function for <format> is called ...")
    # print(rewards)
    return rewards


# customized reward functions: length
def customize_length_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, **kwargs
):
    # schedule length
    if os.getenv("SCHEDULELENGTH", 0) == "1":
        print("SCHEDULELENGTH is set to 1, so schedule max reward for length is used")
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512

    """Reward function that gives higher scores to longer completions."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        if "<think>" not in response or "</think>" not in response:
            rewards.append(min_possible_reward)
            continue
        think_responses = response.split("<think>")[-1].split("</think>")[0].strip()
        reward = round(len(think_responses.split()) / max_reward_len, 2)
        if reward > 1.0:
            reward = 1.0

        final_reward = (
            reward * (max_possible_reward - min_possible_reward) + min_possible_reward
        )
        rewards.append(final_reward)

    # print("\n======= Reward for <length> =======")
    # print("Reward function for <length> is called ...")
    # print(rewards)
    return rewards


def compute_tool_call_reward(
    gt_tools, pd_tools, max_possible_reward, min_possible_reward
):
    if gt_tools == pd_tools:
        # print("Max possible score:", "Exact Match!")
        # print("Score:", max_possible_reward)
        return max_possible_reward

    if os.getenv("COARSEREWARD", 0) == "1":
        print("COARSEREWARD is set to 1, so coarse reward is used")
        if gt_tools != pd_tools:
            return min_possible_reward

    # tool name score
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
            print("INTERMEDIATEREWARD is set to 1, so local max possible is changed")
            local_max_possible += 1.0
        else:
            local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            if str(os.getenv("INTERMEDIATEREWARD", 0)) == "1":
                if gt_tool == pd_tool:
                    best_match = pd_tool
                    best_match_index = i
                    best_match_score = 1.0
                    break
                else:
                    continue

            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            # Calculate correctness score for parameter values
            # correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            # finer_correctness_score:
            correctness_score = 0
            for k, v in gt_params.items():
                if k in pd_params:
                    pd_v = pd_params[k]
                    if type(v) == str and type(pd_v) == str:
                        v = v.replace(",", " ").split(" ")
                        pd_v = pd_v.replace(",", " ").split(" ")
                        correctness_score += match_score(v, pd_v)
                    elif type(v) == int and type(pd_v) == int:
                        correctness_score += 1.0 if v == pd_v else 0.0

            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    # print()
    # print("Max possible score:", local_max_possible)
    # print("Score:", score)

    return (
        max_possible_reward - min_possible_reward
    ) * score / local_max_possible + min_possible_reward


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(
    completions, answer, step, max_possible_reward, min_possible_reward, **kwargs
):
    if str(os.getenv("MAX1STEP30MAX3", 0)) == "1":
        print("MAX1STEP30MAX3 is set to 1, so max 1 -> 30 steps -> max 3")
        if step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3
        else:
            max_possible_reward = max_possible_reward
            min_possible_reward = min_possible_reward

    if str(os.getenv("SCHEDULEREWARD", 0)) == "1":
        print("SCHEDULEREWARD is set to 1, so schedule reward is used")
        max_possible_reward = (max_possible_reward - 2) * step / 150 + 2
        min_possible_reward = (min_possible_reward + 2) * step / 150 - 2
        if max_possible_reward > 3.0:
            max_possible_reward = 3.0
        if min_possible_reward < -3.0:
            min_possible_reward = -3.0

    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer):
        reward = 0.0

        if not check_role("tool_call", ans):
            rewards.append(reward)
            continue

        gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = json.loads(gt_tool_call)

        try:
            pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            pd_tools = json.loads(pd_tools)
            reward = compute_tool_call_reward(
                gt_tools, pd_tools, max_possible_reward, min_possible_reward
            )  # top reward is 2
        except:
            reward = min_possible_reward

        rewards.append(reward)

    return rewards


def compute_score(solution_str, ground_truth, step=0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        predict_str = (
            solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            .split("<|eot_id|>")[0]
            .strip()
        )
    elif "qwen" in exp_name.lower():
        # breakpoint()
        predict_str = (
            solution_str.split("<|im_start|>assistant")[-1]
            .split("<|im_end|>")[0]
            .strip()
        )
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")

    if str(os.getenv("CORRECTMAX1", 0)) == "1":
        print("CORRECTMAX1 is set to 1, so max score is set to 1")
        tool_max_possible = 1.0
        tool_min_possible = -1.0
    else:
        tool_max_possible = 3.0
        tool_min_possible = -3.0

    format_max_possible = 1.0
    format_min_possible = 0.0

    length_max_possible = 1.0
    length_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    fomrat_score = customize_format_reward_func(
        completions, answer, step, format_max_possible, format_min_possible
    )[0]
    correctness_score = customize_correctness_reward_tool(
        completions, answer, step, tool_max_possible, tool_min_possible
    )[0]

    if str(os.getenv("WITHLENGTH", 0)) == "1":
        print("WITHLENGTH is set to 1, so length score is set!")
        length_score = customize_length_reward_func(
            completions, answer, step, length_max_possible, length_min_possible
        )[0]
    else:
        length_score = 0

    score = fomrat_score + correctness_score + length_score

    return score, fomrat_score, correctness_score, length_score
