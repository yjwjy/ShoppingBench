import ujson as json
from pydantic import BaseModel


class BaseTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, str]

    def execute(self, **kwargs):
        raise NotImplementedError()

    def to_string(self):
        return f"Name: {self.name}\nDescription: {self.description}\nParameters: {json.dumps(self.parameters)}"
