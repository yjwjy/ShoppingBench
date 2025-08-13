import re
import ujson as json

from util.message import OUTPUT_ROLES


def format_reward(completion: str, roles: list=[]) -> float:
    if not roles:
        roles = OUTPUT_ROLES

    pos = dict()
    for role in roles:
        start = [m.start() for m in re.finditer(f'<{role}>', completion)]
        end = [m.start() for m in re.finditer(f'</{role}>', completion)]
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
            tool_call_str = completion[pos["tool_call"][0][0] : pos["tool_call"][1][0]].replace("<tool_call>", "").replace("</tool_call>", "")
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


if __name__ == "__main__":
    completion = """..."""
    print(format_reward(completion))
