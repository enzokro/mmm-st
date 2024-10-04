import gc
import atexit
import math
from io import BytesIO
import base64
import time
import logging
from flask import Flask, jsonify, request, Response, render_template, send_file, stream_with_context
from PIL import Image
import torch
import numpy as np
import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download
from fastcore.basics import store_attr
from diffuse import BaseTransformer
app = Flask(__name__)

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    HOST = '0.0.0.0'
    PORT = 8989
    CAP_PROPS = {'CAP_PROP_FPS': 30}
    TRANSFORM_TYPE = "hyper_sd"
    NUM_STEPS = 4  # Adjust this for different inference steps (1-8)
    HEIGHT = 1024
    WIDTH = 1024
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = (WIDTH, HEIGHT)
    NEGATIVE_PROMPT = "low quality, blurry, distorted"
    MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

    # model params
    CFG = 1.0
    STRENGTH = 0.75
    CONTROLNET_SCALE = 0.5
    CONTROLNET_START = 0.0
    CONTROLNET_END = 1.0
    ETA = 1.0
    
torch.manual_seed(Config.SEED)

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(Config.DEVICE)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(Config.DEVICE)
    with torch.no_grad(), torch.autocast(Config.DEVICE):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(Config.HEIGHT, Config.WIDTH),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


class SDXL_Turbo(BaseTransformer):
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
            output_type="pil",
            eta=eta,
        )

        result_image = results.images[0]
        return result_image

class VideoStreamer:
    def __init__(self, device_path='/dev/video0'):
        self.cap = cv2.VideoCapture(device_path)
        if not self.cap.isOpened():
            logger.error("Failed to open video source")
            raise ValueError("Video source cannot be opened")
        for prop, value in Config.CAP_PROPS.items():
            self.cap.set(getattr(cv2, prop), value)

    def get_current_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def release(self):
        self.cap.release()

# Shared global variables
current_prompt = None
previous_frame = None
path = '/dev/video0'
video_streamer = VideoStreamer('/dev/video0') 
image_transformer = SDXL_Turbo(
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

def interpolate_images(image1, image2, alpha=0.5):
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    return Image.blend(image1, image2, alpha)

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

                cnt += 1

                if cnt == decim:
                    img_byte_arr = BytesIO()
                    pil_image = output_frame.resize((1920, 1080))
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