from .base import BaseTool


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = "Terminate the dialogue and declare the task completion status."
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the task.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    def execute(self, **kwargs) -> str:
        return f"The interaction has been completed with status: {kwargs['status']}"
