import asyncio
import math
import uuid

import numpy as np
import websockets
import time
from moviepy.editor import VideoFileClip

import tempfile
import librosa
import soundfile as sf
from PIL import Image
from io import BytesIO
import base64

video_path = '/path/to/video_file'

def image_to_base64(image: Image.Image, fmt='png') -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{fmt};base64,' + base64_str

def audio_to_base64(audio_array, fmt='wav'):
    base64_str = base64.b64encode(audio_array).decode('utf-8')
    return f'data:audio/{fmt};base64,' + base64_str


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr * i:sr * (i + 1)]
        if flatten:
            contents.extend([image, audio])
        else:
            contents.append([image, audio])

    return contents

multi_modal_contents = get_video_chunk_content(video_path)



async def ws_client(url):
    async with websockets.connect(url) as websocket:
        # open session package
        session_id = uuid.UUID
        json = {"capacity_of_str_len": 1000, "session_id": session_id}
        await websocket.send(json),

        # first prompt package
        prompt_package = {
            "messages":[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "描述一下这张图片"
                    }
                ],
            }],
            "max_completion_tokens": 200,
            "temperature":0.0
        }
        await websocket.send(prompt_package)

        # multimodal package
        for image, audio in multi_modal_contents:
            image_base64 = image_to_base64(image)
            audio_base64 = audio_to_base64(audio)

            multi_modal_package = {
                "messages":[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64
                            },
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": audio_base64
                            },
                        },
                    ],
                }],
                "max_completion_tokens": 200,
                "temperature":0.0
            }

            await websocket.send(multi_modal_package)
            time.sleep(1)


        response = await websocket.recv()
        print(response)


asyncio.run(ws_client('ws://localhost:8188'))

