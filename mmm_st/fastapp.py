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
    SharedResources, VideoStreamer, interpolate_images, 
    get_depth_map, get_pose_map, convert_to_pil_image, BaseTransformer
)
from app import SDXL_Hyper

# header for Server-Side Events (SSE)
htmx_sse = Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js")
# our custom javascript
js_script = Script(src="static/app.js")

# for frankenui
tailwind = Script(src="https://cdn.tailwindcss.com?plugins=typography")
frankenui = Link(rel='stylesheet', href='https://unpkg.com/franken-ui@1.1.0/dist/css/core.min.css')
# javascript for frankenui
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

# Initialize shared resources
shared_resources = SharedResources()
shared_resources.image_transformer = SDXL_Hyper(
    num_steps=Config.NUM_STEPS,
    img_size=(Config.WIDTH, Config.HEIGHT)
)

@app.route('/')
def get():
    ui = Div(
        Div(
            Form(
                Input(type='text', id='prompt', name='prompt', placeholder='Enter a new prompt', required='', style='flex: 1;'),
                Button('Submit Prompt', type='submit'),
                id='promptForm',
                style='display: flex; gap: 0px; justify-content: center;',
                hx_post='/set_prompt',
                hx_target='prompt',
                hx_swap='textContent',
                ),
            Form(
                Button('Refresh Scene', type='button'),
                id='refreshForm',
                style='display: flex; gap: 10px; justify-content: center;',
                hx_get='/refresh_scene',
                hx_swap='none',
            ),
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
    while not shared_resources.stop_event.is_set():
        frame = shared_resources.get_frame()
        if frame:
            img_byte_arr = BytesIO()
            frame = convert_to_pil_image(frame)
            frame = frame.resize((Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
            frame.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            res = Img(src=f'data:image/jpeg;base64,{encoded_img}',
                      id='liveImage',
                      alt='Live Image Stream', 
                      style='max-width: 100%; max-height: 100%; object-fit: contain;')
            yield sse_message(res, event="image")

@app.route("/image")
async def get():
    return EventStream(image_events())

# prompt routes
@app.route("/set_prompt")
async def post(prompt: str):
    shared_resources.update_prompt(prompt)
    return prompt

@app.route('/refresh_scene')
def get():
    with shared_resources.lock:
        shared_resources.image_transformer.refresh_latents()
        

def cleanup():
    shared_resources.stop_event.set()
    video_thread.join()
    gc.collect()
    torch.cuda.empty_cache()

atexit.register(cleanup)

serve()