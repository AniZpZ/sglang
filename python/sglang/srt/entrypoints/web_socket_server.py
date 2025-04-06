import json
import multiprocessing as multiprocessing
from typing import AsyncIterator, Callable, Dict, Optional

from sglang import ServerArgs
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.entrypoints.http_server import _GlobalState, open_session, _create_error_response

import asyncio
import websockets

from sglang.srt.managers.io_struct import OpenSessionReqInput, CloseSessionReqInput
from sglang.srt.openai_api.adapter import v1_chat_completions, v1_chat_completions_streaming


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


async def streaming_input_handler(websocket):
    open_session_str = await websocket.recv()
    open_session_data = json.loads(open_session_str)
    open_session_obj = OpenSessionReqInput(**open_session_data)
    try:
        session_id = await _global_state.tokenizer_manager.open_session(open_session_obj, None)
        if session_id is None:
            raise Exception(
                "Failed to open the session. Check if a session with the same id is still open."
            )
    except Exception as e:
        websocket.send(_create_error_response(e))

    while True:
        raw_data = await websocket.recv()
        raw_json = json.loads(raw_data)
        if 'commit' not in raw_json['raw_json']:
            await v1_chat_completions_streaming(_global_state.tokenizer_manager, raw_data)
        else:
            for resp in v1_chat_completions(_global_state.tokenizer_manager, raw_data):
                await websocket.send(resp)
            break



    close_session_obj = CloseSessionReqInput(session_id=open_session_obj.session_id)
    try:
        await _global_state.tokenizer_manager.close_session(close_session_obj, None)
    except Exception as e:
        return _create_error_response(e)



def realtime_launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
        )
    )

    async def run_sglang_websocket_server():
        async with websockets.serve(streaming_input_handler, "localhost", server_args.port):
            await asyncio.Future()

    asyncio.run(run_sglang_websocket_server())

