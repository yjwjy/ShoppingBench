import sys
import multiprocessing
from io import StringIO
from typing import Dict

from toolkit.base import BaseTool

TIMEOUT = 60

DESC = """Executes Python code string."""

CODE_DESC = """Provide Python code to execute. The code must use print() calls to output results."""


class PythonExecute(BaseTool):
    name: str = "python_execute"
    description: str = DESC
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": CODE_DESC,
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            output = output_buffer.getvalue()
            if not output:
                raise Exception("No output produced by the code. Please use print() calls to output results.")
            result_dict["observation"] = output
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    def execute(self, **kwargs) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        code = kwargs.get("code", "")

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(TIMEOUT)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {TIMEOUT} seconds",
                    "success": False,
                }
            return dict(result)
