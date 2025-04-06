"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.entrypoints.web_socket_server import launch_realtime_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    if server_args.enable_multimodal_streaming_input:
        try:
            launch_realtime_server(server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)
    else:
        try:
            launch_server(server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)
