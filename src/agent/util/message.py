import re
import hashlib
import base64
import ujson as json

from pydantic import BaseModel


USER_ROLES = ["user"]
ASSISTANT_ROLES = ["think", "tool_call", "obs", "response"]
OUTPUT_ROLES = ["think", "tool_call", "response"]


def generate_tool_call_id(name: str, parameters: dict, length: int = 8) -> str:
    tool_call_str = tool_call_str = f"{name}\n{parameters}"
    hash_bytes = hashlib.md5(tool_call_str.encode("utf-8")).digest()

    base64_str = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
    clean_str = base64_str.replace('=', '').replace('+', '').replace('/', '')

    return clean_str[:length]


class Message(BaseModel):
    user: str = ""
    think: str = ""
    tool_call: list[dict] = []
    obs: list[dict] = []
    response: str = ""

    def to_dict(self, roles: list[str] = []) -> dict:
        if not roles:
            roles = USER_ROLES + ASSISTANT_ROLES

        result = dict()
        for role in roles:
            if hasattr(self, role) and getattr(self, role):
                result[role] = getattr(self, role)
        return result

    def to_string(self, roles: list[str] = []) -> str:
        if not roles:
            roles = USER_ROLES + ASSISTANT_ROLES

        current = []
        for role in roles:
            if hasattr(self, role) and getattr(self, role):
                content = getattr(self, role)
                if isinstance(content, (dict, list)):
                    content = json.dumps(content)
                elif isinstance(content, str):
                    pass
                else:
                    raise Exception(
                        f"Invalid content type: {type(content)}, content: {content}"
                    )
                current.append(f"<{role}>{content}</{role}>")
        return "\n".join(current)

    def clear(self):
        setattr(self, "user", "")
        setattr(self, "think", "")
        setattr(self, "tool_call", [])
        setattr(self, "obs", [])
        setattr(self, "response", "")

    @classmethod
    def from_dict(clf, message: dict):
        return clf(**message)

    @classmethod
    def from_string(clf, reasoning_content: str, content: str):
        tmp = dict()
        for role in OUTPUT_ROLES:
            matchobj = re.search(f"<{role}>(.+?)</{role}>", content, re.DOTALL)
            if matchobj:
                tmp[role] = matchobj.group(1).strip()
        # think
        if not tmp.get("think") and reasoning_content:
            tmp["think"] = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
        # tool call
        if "tool_call" in tmp:
            tool_call = []
            try:
                json_array = json.loads(tmp["tool_call"])
                if isinstance(json_array, dict):
                    json_array = [json_array]
                for commend in json_array:
                    name = commend["name"]
                    parameters = commend["parameters"]
                    tool_call_id = generate_tool_call_id(name, parameters)
                    tool_call.append({"name": name, "parameters": parameters, "tool_call_id": tool_call_id})
            except:
                pass
            tmp["tool_call"] = tool_call
        return clf(**tmp)


if __name__ == "__main__":
    message = {
        "user": "I want to buy red nike basketball shoes.",
        "think": "The user want to buy red nike basketball shoes, I should use the find_product tool.",
        "tool_call": [
            {
                "name": "find_product",
                "parameters": {"q": "red nike basketball shoes", "page": 1},
            }
        ],
    }
    m = Message.from_dict(message)
    print(m.user)
    print(m.think)
    print(m.tool_call)
    print(m.to_dict())
    print(m.to_string())
