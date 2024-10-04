import gc
import time
import math
import atexit
import base64
import logging
from io import BytesIO
from PIL import Image
from flask import Flask, jsonify, request, Response, render_template, send_file, stream_with_context
import cv2
import torch
import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download
from fastcore.basics import store_attr

# our imports
from config import Config
from utils import BaseTransformer, get_depth_map, VideoStreamer, interpolate_images, convert_to_pil_image

# create the flask app
app = Flask(__name__)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set seed
torch.manual_seed(Config.SEED)

class SDXL_Hyper(BaseTransformer):
    def __init__(
            self,
            cfg=Config.CFG,
            strength=Config.STRENGTH,
            controlnet_scale=Config.CONTROLNET_SCALE,
            controlnet_start=Config.CONTROLNET_START,
            controlnet_end=Config.CONTROLNET_END,
            width=Config.WIDTH,
            height=Config.HEIGHT,
            **kwargs,
        ):
        super().__init__(**kwargs)
        store_attr()

        self.generator = torch.Generator(device="cpu").manual_seed(Config.SEED)
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-lora.safetensors"

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)

        self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipe.fuse_lora()

        # Use TCD scheduler for better image quality
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

        if self.device != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.controlnet.to(memory_format=torch.channels_last)

        # Initialize latents
        batch_size = 1
        num_channels_latents = self.pipe.unet.config.in_channels
        self.latents_shape = (batch_size, num_channels_latents, self.height // self.pipe.vae_scale_factor, self.width // self.pipe.vae_scale_factor)
        self.latents = self.refresh_latents()

    def transform(self, image, prompt) -> Image.Image:
        image = self.input_image or image
        negative_prompt = self.negative_prompt
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        control_image = get_depth_map(image)

        # Lower eta for more detail in multi-step inference
        eta = Config.ETA

        results = self.pipe(
            prompt=prompt,
            image=image,
            control_image=control_image,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=self.generator,
            num_inference_steps=self.num_steps,
            guidance_scale=self.cfg,
            strength=self.strength,
            controlnet_conditioning_scale=self.controlnet_scale,
            control_guidance_start=self.controlnet_start,
            control_guidance_end=self.controlnet_end,
            width=self.width,
            height=self.height,
            latents=self.get_latents(),
            output_type="pil",
            eta=eta,
        )

        result_image = results.images[0]
        return result_image

    def get_latents(self):
        return self.latents.clone()

    def refresh_latents(self):
        self.latents = torch.randn(self.latents_shape, generator=self.generator, device=self.device, dtype=Config.DTYPE)
        return self.latents



# Shared global variables
current_prompt = None
previous_frame = None
video_streamer = VideoStreamer(Config.VIDEO_PATH) 
image_transformer = SDXL_Hyper(
    num_steps=Config.NUM_STEPS,
    img_size=(Config.WIDTH, Config.HEIGHT)
)

def transform_frame(previous_frame, prompt=None):
    global video_streamer
    global image_transformer
    current_frame = video_streamer.get_current_frame()
    if current_frame is None:
        return previous_frame

    if prompt:
        transformed_image = image_transformer(current_frame, prompt)
    else:
        transformed_image = current_frame

    return transformed_image


@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/set_prompt', methods=['POST'])
def set_prompt():
    global current_prompt
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    current_prompt = prompt
    return jsonify({"message": "Prompt set successfully"})

@app.route('/stream')
def stream():
    def generate():
        global current_prompt
        global video_streamer
        global image_transformer
        global previous_frame

        cnt, decim = 0, 2
        try:
            while True:
                output_frame = transform_frame(
                    previous_frame, 
                    prompt=current_prompt)
                previous_frame = output_frame

                # # set the new image as the previous, to condition on
                # image_transformer.set_image(previous_frame)

                cnt += 1

                if cnt == decim:
                    img_byte_arr = BytesIO()
                    pil_image = convert_to_pil_image(output_frame)
                    pil_image = output_frame.resize((Config.OUTPUT_WIDTH, Config.OUTPUT_HEIGHT))
                    pil_image.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    yield f"data: {encoded_img}\n\n"
                    cnt = 0
        finally:
            video_streamer.release()

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def cleanup():
    print("Application is shutting down. Cleaning up resources.")
    global video_streamer
    global image_transformer
    video_streamer.release()
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    atexit.register(cleanup)
    app.run(
        host=Config.HOST, 
        port=Config.PORT,
        debug=True, 
        threaded=True, 
        use_reloader=False,
    )