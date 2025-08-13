import os
import sys
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process


model_path = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

tp = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
server_process, port = launch_server_cmd(f"python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --mem-fraction-static 0.8 --tp {tp}")

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")