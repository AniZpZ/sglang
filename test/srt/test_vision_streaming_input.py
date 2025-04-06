import asyncio
import websockets
import time


async def ws_client(url):
    for i in range(1, 40):
        async with websockets.connect(url) as websocket:
            await websocket.send("Hello, I am PyPy.")
            response = await websocket.recv()
        print(response)
        time.sleep(1)

asyncio.run(ws_client('ws://localhost:8188'))

