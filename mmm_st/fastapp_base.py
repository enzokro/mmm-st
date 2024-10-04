import gc
import json
import time
import traceback
from fasthtml.common import *
import base64
import atexit
from io import BytesIO
import torch
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    TCDScheduler
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from config import Config
from utils import (
    VideoStreamer, interpolate_images, 
    get_depth_map, get_pose_map, convert_to_pil_image, BaseTransformer
)
from app import SDXL_Hyper

# header for Server-Side Events (SSE) and our custom javascript
htmx_sse = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
js_script = Script(src="static/app.js")

# frankenui
tailwind = Script(src="https://cdn.tailwindcss.com?plugins=typography")
frankenui = Link(rel='stylesheet', href='https://unpkg.com/franken-ui@1.1.0/dist/css/core.min.css')
fjs_1 = Script(src="https://unpkg.com/franken-ui@1.1.0/dist/js/core.iife.js", type="module")
fjs_2 = Script(src="https://unpkg.com/franken-ui@1.1.0/dist/js/icon.iife.js", type="module")

# group up the headers
headers = [
    htmx_sse,
    js_script,
    tailwind,
    frankenui,
    fjs_1,
    fjs_2,
]

# create the app
app, _ = fast_app(
    debug=True,
    hdrs=headers,
    ws_hdr=False,
)

# Shared global variables
current_prompt = None
previous_frame = None
video_streamer = VideoStreamer(Config.VIDEO_PATH) 
image_transformer = SDXL_Hyper(
    num_steps=Config.NUM_STEPS,
    img_size=(Config.WIDTH, Config.HEIGHT)
)


@app.route('/')
def get():
    ui = Div(
        Div(
            Form(
                Input(type='text', id='prompt', name='prompt', placeholder='Enter a new prompt', required='', style='flex: 1;'),
                Button('Submit Prompt', type='submit', cls="uk-button uk-button-primary"),
                id='promptForm',
                style='justify-content: center;',
                hx_post='/set_prompt',
                hx_target='prompt',
                hx_swap='textContent',
                ),
            Form(
                Button('Refresh Scene', type='button', cls="uk-button uk-button-default"),
                id='refreshForm',
                style='justify-content: center;',
                hx_get='/refresh_scene',
                hx_swap='none',
            ),
            cls="flex flex-row flex-grow",
            id='control-section',
        ),
        Div(
            Img(id='liveImage', src='', alt='Live Image Stream', style='max-width: 100%; max-height: 100%; object-fit: contain;'),
            id='stream-section',
            style='height: 90%; width: 100%; display: flex; justify-content: center; align-items: center;',
            hx_ext='sse',
            sse_connect='/image',
            sse_swap='image',
            hx_swap='innerHTML',
        ),
         style='margin: 0; padding: 0; overflow: hidden; height: 100vh; width: 100vw;'
    )
    return ui

# SSE routes
async def image_events():
    global current_prompt
    global video_streamer
    global image_transformer
    global previous_frame

    cnt, decim = 0, 2
    try:
        while True:
            # get the current frame
            current_frame = video_streamer.get_current_frame()
            if current_frame is None:
                output_frame = previous_frame

            # apply the prompt if it exists
            if current_prompt:
                output_frame = image_transformer(current_frame, current_prompt)
            else:
                output_frame = current_frame
            # update the previous frame
            previous_frame = output_frame

            # # set the new image as the previous, to condition on
            # image_transformer.set_image(previous_frame)

            cnt += 1

            if cnt == decim:
                img_byte_arr = BytesIO()
                # pil_image = convert_to_pil_image(output_frame)
                pil_image = output_frame.resize((Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT))
                pil_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                res = Img(src=f'data:image/jpeg;base64,{encoded_img}',
                    id='liveImage',
                    alt='Live Image Stream', 
                    style='max-width: 100%; max-height: 100%; object-fit: contain;')
                yield sse_message(res, event="image")
    finally:
        video_streamer.release()
        

@app.route("/image")
async def get():
    "Yields image events"
    return EventStream(image_events())

# prompt routes
@app.route("/set_prompt")
async def post(prompt: str):
    global current_prompt
    if prompt:
        current_prompt = prompt
    return prompt

@app.route('/refresh_scene')
async def get():
    global image_transformer
    image_transformer.refresh_latents()
    return P("ok")
        

def cleanup():
    shared_resources.stop_event.set()
    video_thread.join()
    gc.collect()
    torch.cuda.empty_cache()

atexit.register(cleanup)

serve()