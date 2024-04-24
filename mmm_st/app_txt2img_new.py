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
from mmm_st.config import Config as BaseConfig
from mmm_st.diffuse import BaseTransformer
from mmm_st.video import convert_to_pil_image
from fastcore.basics import store_attr

# tests with SDXL Lightning
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

app = Flask(__name__)

# Import and use configurations from an external module if necessary
class Config:
    HOST = '0.0.0.0'
    PORT = 8989
    CAP_PROPS = {'CAP_PROP_FPS': 30}
    NUM_STEPS = 4
    HEIGHT = 1024
    WIDTH = 1024
    SEED  = BaseConfig.SEED


# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the seed and data type for torch
SEED = Config.SEED
torch.manual_seed(Config.SEED)
torch_dtype = torch.float16

# Models for image and controlnet
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
unet_id = "ByteDance/SDXL-Lightning"
unet_ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# controlnet_model = "diffusers/controlnet-canny-sdxl-1.0"
controlnet_model = "diffusers/controlnet-depth-sdxl-1.0"


# for depth estimation
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
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
            cfg=1.0,
            strength=0.75,
            canny_low_threshold=100,
            canny_high_threshold=200,
            controlnet_scale=0.5,
            controlnet_start=0.0,
            controlnet_end=1.0,
            width=Config.WIDTH,
            height=Config.HEIGHT,
            **kwargs,
        ):
        super().__init__(**kwargs)
        store_attr()
        self.generator = torch.Generator(device="cpu").manual_seed(SEED)
        self.device = torch.device(self.device)

        # initialize the control net model
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        ).to(self.device)

        # load in the VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch_dtype
        )

        # Load the model with a controlnet.
        unet_config = UNet2DConditionModel.load_config(model_id, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch_dtype)
        unet.load_state_dict(load_file(hf_hub_download(unet_id, unet_ckpt), device="cuda"))
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
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
        self.pipe.to(device=self.device, dtype=torch_dtype).to(self.device)
        if self.device != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.latents = self.make_latents()


    def make_latents(self):
        # create uniform, shared latents
        batch_size, num_images_per_prompt = 1, 1
        batch_size *= num_images_per_prompt
        num_channels_latents = self.pipe.unet.config.in_channels
        shape = (batch_size, num_channels_latents, self.height // self.pipe.vae_scale_factor, self.width // self.pipe.vae_scale_factor)
        latents = randn_tensor(shape, generator=self.generator, device=self.device, dtype=torch_dtype)
        # print('local latents:', latents.shape, latents.dtype, latents.device, latents.max().item(), latents.min().item(), latents.mean().item())
        return latents
    
    def refresh_latents(self):
        self.latents = self.make_latents()

    def get_latents(self):
        return self.latents.clone()

    def _initialize_pipeline(self):
        pass

    def transform(self, image, prompt) -> Image.Image:

        # work on either the given image or the stored `input_image`
        image = self.input_image or image

        # get the controlnet image
        control_image = get_depth_map(image)
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


class VideoStreamer:
    """ Continuously reads frames from a video capture source. """
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
    img_size = (Config.WIDTH, Config.HEIGHT)
)


def transform_frame(previous_frame, prompt=None):
    """
    Fetches a frame from the video streamer, applies a transformation based on the provided prompt,
    and interpolates it with the previous frame.
    
    Args:
        video_streamer (VideoStreamer): The video streamer object.
        image_transformer (Callable): Function to transform the frame based on the prompt.
        previous_frame (Image.Image): The previous frame to interpolate with.
        prompt (str, optional): The prompt based on which the transformation is applied.

    Returns:
        Image.Image: The updated frame after transformation and interpolation.
    """
    global video_streamer
    global image_transformer
    current_frame = video_streamer.get_current_frame()
    if current_frame is None:
        return previous_frame  # Return the previous frame if no new frame is captured

    # if previous_frame is not None:
    #     current_frame = interpolate_images(current_frame, previous_frame, 0.25)

    if prompt:
        # Transform the current frame based on the prompt
        transformed_image = image_transformer(current_frame, prompt)
    else:
        transformed_image = current_frame

    # Interpolate between the previous frame and the transformed image
    if previous_frame is not None:
        output_frame = interpolate_images(previous_frame, transformed_image, alpha=0.75)
    else:
        output_frame = transformed_image

    output_frame = transformed_image

    return output_frame


def interpolate_images(image1, image2, alpha=0.5):
    """ Interpolates two images with a given alpha blending factor. """
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


@app.route('/refresh_prompt', methods=['POST'])
def refresh_latents():
    global image_transformer
    try:
        prompt = request.json.get('signal')  # Ensure we get the expected signal
        if prompt != "refresh":  # Basic check for the correct signal
            return jsonify({"error": "Invalid signal"}), 400
        image_transformer.refresh_latents()  # Refresh the state as needed
        return jsonify({"message": "Scene refreshed"})
    except Exception as e:  # Catch potential errors and log them
        print(f"Error refreshing scene: {e}")
        return jsonify({"error": "Could not refresh scene"}), 500


@app.route('/stream')
def stream():
    def generate():
        global current_prompt
        global video_streamer
        global image_transformer
        global previous_frame

        cnt, decim = 0, 1
        try:
            while True:
                output_frame = transform_frame(
                    previous_frame, 
                    prompt=current_prompt)
                previous_frame = output_frame

                cnt += 1

                # # set the new image as the previous, to condition on
                # image_transformer.set_image(previous_frame)

                if cnt == decim:
                    img_byte_arr = BytesIO()
                    pil_image = convert_to_pil_image(output_frame)
                    # rescale to better fit the image
                    pil_image = pil_image.resize((1920, 1080))
                    pil_image.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    yield f"data: {encoded_img}\n\n"
                    cnt = 0
                # time.sleep(1 / 15)  # Control frame rate
        finally:
            video_streamer.release()

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def cleanup():
    print("Application is shutting down. Cleaning up resources.")
    global video_streamer
    global image_transformer
    # del image_transformer
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
