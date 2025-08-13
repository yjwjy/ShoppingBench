import os
import sys
import time
import copy
import portalocker
import multiprocessing as mp
import asyncio
import ujson as json
from tqdm import tqdm

from toolkit import tools, toolmap
from util.llm import ask_llm
from util.message import Message, USER_ROLES, ASSISTANT_ROLES


MAX_STEPS = 30


def get_system_prompt(config: dict) -> str:
    with open(config["system_prompt_file"], "r") as fin:
        prompt_template = fin.read().strip()

    description = []
    for i, tool in enumerate(tools):
        if tool.name in config.get("exclude_tools", []):
            continue
        description.append(f"{i+1}. {tool.to_string()}")
    toolkit_description = "\n\n".join(description)

    return prompt_template.replace("<|toolkit_description|>", toolkit_description)


def get_user_prompt(message: Message, history_messages: list[str]) -> str:
    user_message = message.to_string(USER_ROLES)
    if user_message:
        history_messages.append(user_message)

    assistant_message = message.to_string(ASSISTANT_ROLES)
    if assistant_message:
        history_messages.append(assistant_message)

    history = "\n\n".join(history_messages)
    return f"# Dialogue Records History\n{history}"


def think(
    system_prompt: str,
    user_prompt: str,
    model_config: dict,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, Message]:
    reasoning_content, content = ask_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_config=model_config,
        base_url=base_url,
        api_key=api_key
    )

    return reasoning_content, content, Message.from_string(reasoning_content, content)


def act(message: Message) -> list[dict]:
    obs = []
    for commend in message.tool_call:
        if commend["name"] not in toolmap:
            continue
        tool = toolmap[commend["name"]]
        obs.append(
            {
                "tool_call_id": commend["tool_call_id"],
                "results": asyncio.run(tool.execute(**commend["parameters"])) if tool.name == "web_search" else tool.execute(**commend["parameters"]),
            }
        )
    return obs


def is_terminate(message: Message) -> bool:
    if (not message.think and not message.tool_call and not message.response) or "terminate" in { commend["name"] for commend in message.tool_call }:
        return True
    return False


def react_loop(query: str, config: dict):
    corpus_tracker = []
    history_messages = []
    message = Message(user=query)
    system_prompt = get_system_prompt(config)
    #print(f"System Prompt:\n{system_prompt}")
    for step in range(1, MAX_STEPS + 1):
        user_prompt = get_user_prompt(message, history_messages)
        message.clear()
        reasoning_content, content, message = think(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_config=config["model_config"],
            base_url=config.get("base_url", ""),
            api_key=config.get("api_key", ""),
        )
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
        #print(f"{'*' * 20}Setps: {step}/{MAX_STEPS}{'*' * 20}\nReasoning Content: {reasoning_content}\nContent: {content}\nMessage: {json.dumps(message.to_dict(), indent=4)}\n")
        if is_terminate(message):
            break

    with open(config["rollout_file"], "a") as fout:
        portalocker.lock(fout, portalocker.LOCK_EX)
        fout.write(f"{json.dumps(corpus_tracker)}\n")
        fout.flush()
        portalocker.unlock(fout)


def producer(queue: mp.Queue, config: dict):
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
            queue.put(query)
            had_queries.add(query)
            pbar.update(1)
            #print(f"Put query: {query}")
    queue.put(None)


def consumer(queue: mp.Queue, config: dict):
    while True:
        query = queue.get()
        if query is None:
            queue.put(None)
            break
        #print(f"Get query: {query}")
        react_loop(query, config)


def rollout(config: dict):
    queue = mp.Queue(config["threads"])

    # Create processes
    producer_process = mp.Process(target=producer, args=(queue, config))
    consumers = []
    for _ in range(config["threads"]):
        consumers.append(mp.Process(target=consumer, args=(queue, config)))

    # Start processes
    producer_process.start()
    for consumer_process in consumers:
        consumer_process.start()

    # Join processes
    producer_process.join()
    for consumer_process in consumers:
        consumer_process.join() 


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as fin:
        config = json.load(fin)
    if config["task"] != "knowledge":
        config["exclude_tools"] = config.get("exclude_tools", []) + ["web_search"]
    rollout(config)
