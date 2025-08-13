import os
import sys
import time
import copy
import portalocker
import asyncio
import ujson as json
from tqdm import tqdm
from colorama import init, Fore

from util.message import Message
from toolkit import tools, toolmap
from run_rollout import (
    get_system_prompt,
    get_user_prompt,
    is_terminate,
    MAX_STEPS,
)


init(autoreset=True)


def think(
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str, Message]:
    print(Fore.RED + f"=== System Prompt ===\n{system_prompt}\n\n")
    print(Fore.BLUE + f"=== User Prompt ===\n{user_prompt}\n\n")
    while True:
        tools_str = "\n".join(f"{i}. {tool.name}" for i, tool in enumerate(tools))
        name = input(Fore.GREEN + f"{tools_str}\nPlease input tool name or serial number: ")
        if name.isdigit():
            name = tools[int(name)].name

        keys = []
        if name == "find_product":
            keys = ["q", "page", "shop_id", "price", "sort", "service"]
        elif name in {"view_product_information", "recommend_product"}:
            keys = ["product_ids"]
        elif name == "terminate":
            keys = ["status"]
        elif name == "python_execute":
            keys = ["code"]
        elif name == "web_search":
            keys = ["q", "max_results"]
        else:
            print(Fore.YELLOW + f"Invalid tool name: {name}, please input again.")

        if keys:
            parameters = dict()
            for key in keys:
                value = input(Fore.GREEN + f"Please input {key}: ")
                if value:
                    parameters[key] = value
            if parameters:
                break
            else:
                print(Fore.YELLOW + f"Parameters is empty, please input again.")

    tool_call = [{"name": name, "parameters": parameters}]
    return "", "", Message.from_dict({"tool_call": tool_call})


def act(message: Message):
    obs = []
    for commend in message.tool_call:
        name = commend["name"]
        parameters = commend["parameters"]
        tool_call_id = commend["tool_call_id"]

        if name not in toolmap:
            continue

        tool = toolmap[name]
        results = asyncio.run(tool.execute(**parameters)) if tool.name == "web_search" else tool.execute(**parameters)
        if name == "find_product" and results:
            product_ids = []
            for product in results:
                product_ids.append(product["product_id"])

        observation = {
            "name": name,
            "parameters": parameters,
            "tool_call_id": tool_call_id,
            "results": results,
        }
        if name == "find_product" and results and product_ids:
            observation["prpduct_ids"] = ",".join(product_ids)
        obs.append(observation)
    return json.dumps(obs, indent=2)


def react_loop(query: str, config: dict):
    corpus_tracker = []
    history_messages = []
    message = Message(user=query)
    system_prompt = get_system_prompt(config)
    for step in range(1, MAX_STEPS + 1):
        user_prompt = get_user_prompt(message, history_messages)
        message.clear()
        reasoning_content, content, message = think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        print(message)
        if message.tool_call:
            message.obs = act(message)

        corpus_tracker.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "completion": {
                    "reasoning_content": reasoning_content,
                    "content": content,
                    "message": copy.deepcopy(message.to_dict()),
                },
                "extra_info": {
                    "step": step,
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                },
            }
        )
        if is_terminate(message):
            break

    with open(config["rollout_file"], "a") as fout:
        portalocker.lock(fout, portalocker.LOCK_EX)
        fout.write(f"{json.dumps(corpus_tracker)}\n")
        fout.flush()
        portalocker.unlock(fout)


def rollout(config: dict):
    had_queries = set()
    if os.path.exists(config["rollout_file"]):
        with open(config["rollout_file"], "r") as fin:
            portalocker.lock(fin, portalocker.LOCK_EX)
            for line in fin:
                jsonobj = json.loads(line.strip())
                query = jsonobj[0]["extra_info"]["query"]
                had_queries.add(query)
            portalocker.unlock(fin)

    total = int(os.popen(f"wc -l {config['synthesize_file']}").read().strip().split(" ", 1)[0])
    pbar = tqdm(total=total - len(had_queries), desc="Start rolling out the remaining queries: ")
    with open(config["synthesize_file"], "r") as fin:
        for line in fin:
            jsonobj = json.loads(line.strip())
            query = jsonobj["query"]
            if query in had_queries:
                continue
            react_loop(query, config)
            had_queries.add(query)
            pbar.update(1)


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    rollout(config)
