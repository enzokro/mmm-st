import gc
import math
import time
import base64
import atexit
import logging
import threading
from io import BytesIO
import numpy as np
import torch
import cv2
from PIL import Image
from fastcore.basics import store_attr
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import DPTImageProcessor, DPTForDepthEstimation
from huggingface_hub import hf_hub_download
from flask import Flask, jsonify, request, Response, render_template, stream_with_context
from mmm_st.config import Config as BaseConfig
from mmm_st.diffuse import BaseTransformer
from mmm_st.video import convert_to_pil_image


# Create Flask app
app = Flask(__name__)

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application configuration
class Config:
    HOST = '0.0.0.0'
    PORT = 8989
    CAP_PROPS = {}
    NUM_STEPS = 4
    HEIGHT = 1024
    WIDTH = 1024
    SEED  = BaseConfig.SEED
    VIDEO_PATH = '/dev/video0'

    # to display the final output frame on stream
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080

    # alpha blending factor for frame interpolation
    FRAME_BLEND = 0.75

    # Models for image and controlnet
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    UNET_ID = "ByteDance/SDXL-Lightning"
    UNET_CKPT = "sdxl_lightning_4step_unet.safetensors"
    CONTROLNET = "diffusers/controlnet-depth-sdxl-1.0"

    # Default data type for torch
    DTYPE = torch.float16

# Ensure torch uses the correct seed
torch.manual_seed(Config.SEED)


# Class for managing shared resources
class SharedResources:
    def __init__(self):
        self.image_transformer = SDXL(
            num_steps=Config.NUM_STEPS,
            height=Config.HEIGHT,
            width=Config.WIDTH,
        )
        self.current_frame = None
        self.current_prompt = None
        self.lock = threading.Lock()  # Ensure thread-safe access
        self.stop_event = threading.Event()  # Signal to stop thread

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def update_prompt(self, prompt):
        with self.lock:
            self.current_prompt = prompt

    def get_prompt(self):
        with self.lock:
            return self.current_prompt


class VideoStreamer:
    "Continuously reads frames from a video capture source."
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


# Video processing class that handles the transformation and streaming
class VideoProcessingThread(threading.Thread):
    def __init__(self, shared_resources, device_path=Config.VIDEO_PATH):
        super().__init__(daemon=True)
        self.shared_resources = shared_resources
        self.video_streamer = VideoStreamer(device_path)
        self.previous_frame = None
        self.stop_event = shared_resources.stop_event

    def run(self):
        def interpolate_images(image1, image2, alpha=0.5):
            """ Interpolates two images with a given alpha blending factor. """
            if image1.size != image2.size:
                image2 = image2.resize(image1.size)
            return Image.blend(image1, image2, alpha)

        while not self.stop_event.is_set():
            current_frame = self.video_streamer.get_current_frame()
            if current_frame is None:
                continue

            current_prompt = self.shared_resources.get_prompt()

            # Apply transformation based on the current prompt
            transformed_frame = self.shared_resources.image_transformer.transform(
                current_frame,
                current_prompt,
            )

            # Interpolate and update the shared frame
            if self.previous_frame is not None:
                final_frame = interpolate_images(self.previous_frame, transformed_frame, Config.FRAME_BLEND)
            else:
                final_frame = transformed_frame

            self.shared_resources.update_frame(final_frame)
            self.previous_frame = final_frame

            # final sanity break
            if self.stop_event.is_set(): break

        self.video_streamer.release()  # Release resources when stopping


class SDXL(BaseTransformer):
    def __init__(
            self,
            cfg=1.0,
            strength=0.75,
            canny_low_threshold=100,
            canny_high_threshold=200,
            controlnet_scale=0.5,
            controlnet_start=0.0,
            controlnet_end=1.0,
            width=Config.WIDTH,
            height=Config.HEIGHT,
            dtype=Config.DTYPE,
            **kwargs,
        ):
        store_attr()
        self.generator = torch.Generator(device="cpu").manual_seed(Config.SEED)
        self.device = torch.device(self.device)
        super().__init__(**kwargs)

    def _initialize_pipeline(self):

        # initialize the control net model
        controlnet = ControlNetModel.from_pretrained(
            Config.CONTROLNET,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)

        # for depth estimation
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
        self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

        # load in the VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype
        )

        # Load the Lightning UNet
        unet_config = UNet2DConditionModel.load_config(Config.MODEL_ID, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config).to(self.device, self.dtype)
        unet.load_state_dict(load_file(hf_hub_download(Config.UNET_ID, Config.UNET_CKPT), device=self.device))
        
        # Load the model with a controlnet.
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            Config.MODEL_ID,
            use_safetensors=True,
            unet=unet,
            vae=vae,
            # NOTE: try using the same contorlnet twice, with different strengths and images
            controlnet=controlnet,
        )

        # Ensure sampler uses "trailing" timesteps.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        # final pipeline setup
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=self.device, dtype=self.dtype).to(self.device)
        if self.device not in ("mps", torch.device("mps")):
            self.pipe.unet.to(memory_format=torch.channels_last)

        # start with a fixed set of latents
        self.latents = self._make_latents()


    def _make_latents(self):
        "Create a fixed set of latents for reproducible starting images."
        batch_size, num_images_per_prompt = 1, 1
        batch_size *= num_images_per_prompt
        num_channels_latents = self.pipe.unet.config.in_channels
        shape = (batch_size, num_channels_latents, self.height // self.pipe.vae_scale_factor, self.width // self.pipe.vae_scale_factor)
        latents = randn_tensor(shape, generator=self.generator, device=self.device, dtype=self.dtype)
        return latents
    
    def refresh_latents(self):
        "Grabs a new set of latents to refresh the starting image generation."
        self.latents = self._make_latents()

    def get_latents(self):
        "Returns a copy of the current latents, to prevent them from being overriden."
        return self.latents.clone()

    def transform(self, image, prompt) -> Image.Image:
        """Transforms the given `image` based on the `prompt`.
        
        Uses a controlnet to condition and manage the generation. 
        """

        # get the controlnet image
        control_image = self.get_depth_map(image)
        steps = self.num_steps
        if int(steps * self.strength) < 1:
            steps = math.ceil(1 / max(0.10, self.strength))

        # generate and return the image
        results = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            image=control_image,
            num_inference_steps=steps,
            guidance_scale=self.cfg,
            strength=self.strength,
            controlnet_conditioning_scale=self.controlnet_scale,
            control_guidance_start=self.controlnet_start,
            control_guidance_end=self.controlnet_end,
            width=self.width,
            height=self.height,
            output_type="pil",
            generator=self.generator,
            # add the fixed, starting latents
            latents=self.get_latents(),
        )
        result_image = results.images[0]
        return result_image
    
    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(self.height, self.width),
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
    

# Initialize shared resources and video processing thread
shared_resources = SharedResources()
video_thread = VideoProcessingThread(shared_resources)
video_thread.start()


# Flask endpoint to serve the index page
@app.route('/')
def index():
    return render_template('index.html')


# Flask endpoint to set the prompt
@app.route('/set_prompt', methods=['POST'])
def set_prompt():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    shared_resources.update_prompt(prompt)  # Update the prompt
    return jsonify({"message": "Prompt set successfully"})


# Flask endpoint to refresh the latents
@app.route('/refresh_latents', methods=['POST'])
def refresh_latents():
    try:
        signal = request.json.get('signal')
        if signal != "refresh":
            return jsonify({"error": "Invalid signal"}), 400
        
        # Refresh the latents
        with shared_resources.lock:
            shared_resources.image_transformer.refresh_latents()
        
        return jsonify({"message": "Latents refreshed"})
    except Exception as e:
        logger.error(f"Error refreshing latents: {e}")
        return jsonify({"error": "Could not refresh latents"}), 500


# Flask endpoint to stream the latest frame
@app.route('/stream')
def stream():
    def generate():
        while not shared_resources.stop_event.is_set():
            frame = shared_resources.get_frame()
            if frame is not None:
                frame = convert_to_pil_image(frame)
                # rescale to better fit the image
                frame = frame.resize((Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
                img_byte_arr = BytesIO()
                frame.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                yield f"data: {encoded_img}\n\n"
                # time.sleep(1 / 30)  # Frame rate control

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# Cleanup function for releasing resources
def cleanup():
    shared_resources.stop_event.set() 
    video_thread.join() 
    gc.collect()  
    torch.cuda.empty_cache()  

# Register cleanup function at exit
atexit.register(cleanup)

if __name__ == "__main__":
    app.run(
        host=Config.HOST, 
        port=Config.PORT,
        debug=True, 
        threaded=True, 
        use_reloader=False,
    )
