import os
import time
import logging

from openai import OpenAI


MAX_RETRIES = 10

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()


def chat_completion_stream(client: OpenAI, messages: list[dict[str, str]], model_config: dict):
    stream = client.chat.completions.create(
        messages=messages,
        extra_headers={"Accept": "text/event-stream"},
        **model_config,
    )

    reasoning_content = ""
    content = ""
    for event in stream:
        try:
            reasoning_content += event.choices[0].delta.reasoning_content
        except:
            pass
        try:
            content += event.choices[0].delta.content
        except:
            pass

    return reasoning_content, content


def chat_completion(client: OpenAI, messages: list[dict[str, str]], model_config: dict):
    completion = client.chat.completions.create(
        messages=messages,
        extra_headers={"Accept": "text/event-stream"},
        **model_config,
    )

    reasoning_content = ""
    content = ""
    try:
        reasoning_content = completion.choices[0].message.reasoning_content
    except:
        pass
    try:
        content = completion.choices[0].message.content
    except:
        pass

    return reasoning_content, content


def ask_llm(
    messages: list[dict[str, str]],
    model_config: dict,
    base_url: str = None,
    api_key: str = None,
) -> tuple[str, str]:
    success = False
    for i in range(MAX_RETRIES):
        try:
            client = OpenAI(
                base_url=base_url if base_url else os.environ.get("OPENAI_BASE_URL"),
                api_key=api_key if api_key else os.environ.get("OPENAI_API_KEY"),
            )

            if model_config.get("stream", False):
                reasoning_content, content = chat_completion_stream(
                    client, messages, model_config
                )
            else:
                reasoning_content, content = chat_completion(
                    client, messages, model_config
                )

            if reasoning_content or content:
                success = True
                break
            else:
                raise Exception("reasoning_content and content is empty")
        except Exception as e:
            logger.error(f"Error occurred: {e}. Retry {i+1}/{MAX_RETRIES}.")
            time.sleep(3)
        finally:
            client.close()

    if not success:
        logger.error(f"Retry {MAX_RETRIES} but can't success!")
        reasoning_content = ""
        content = ""
    return reasoning_content, content


if __name__ == "__main__":
    reasoning_content, content = ask_llm(
        messages=[{"role": "user", "content": "hi"}],
        model_config={
            "model": "gemini-2.5-flash",
            "temperature": 0,
            "max_tokens": 8192,
            "extra_body": {
                "google": {
                    "thinkingConfig": {
                        "includeThoughts": True
                    },
                    "thought_tag_marker": "think"
                }
            }
        },
    )
    print(f"reasoning_content: {reasoning_content}\ncontent: {content}")
