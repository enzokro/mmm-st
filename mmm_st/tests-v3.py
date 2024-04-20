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
import cv2  # Using OpenCV for video capture
from transformers import DPTImageProcessor, DPTForDepthEstimation
from mmm_st.config import Config
from mmm_st.diffuse import BaseTransformer
from mmm_st.video import convert_to_pil_image
from fastcore.basics import store_attr
import matplotlib
import matplotlib.cm


torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval()
model_zoe_n = model_zoe_n.to("cuda")


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    # gamma correction
    img = img / 255
    img = np.power(img, 2.2)
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img


def get_zoe_depth_map(image):
    with torch.autocast("cuda", enabled=True):
        depth = model_zoe_n.infer_pil(image)
    depth = colorize(depth, cmap="gray_r")
    return depth

### Tests with SDXL turbo
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    AutoencoderTiny,
)

app = Flask(__name__)

SEED = Config.SEED
torch.manual_seed(Config.SEED)

# Import and use configurations from an external module if necessary
class Config:
    HOST = '0.0.0.0'
    PORT = 8989
    CAP_PROPS = {'CAP_PROP_FPS': 30}
    TRANSFORM_TYPE = "kandinsky"
    NUM_STEPS = 4
    HEIGHT = 1024
    WIDTH = 1024


# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# controlnet_model = "diffusers/controlnet-canny-sdxl-1.0"
# controlnet_model = "diffusers/controlnet-depth-sdxl-1.0"
controlnet_model = "diffusers/controlnet-zoe-depth-sdxl-1.0"
model_id = "stabilityai/sdxl-turbo"
taesd_model = "madebyollin/taesdxl"

torch_dtype = torch.float16

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
            cfg=1.5,
            strength=1.0,
            canny_low_threshold=100,
            canny_high_threshold=200,
            controlnet_scale=0.8,
            controlnet_start=0.0,
            controlnet_end=1.0,
            width=Config.WIDTH,
            height=Config.HEIGHT,
            **kwargs,
        ):
        super().__init__(**kwargs)
        store_attr()

        self.generator = torch.Generator(device="cpu").manual_seed(SEED)
        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        ).to(self.device)
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype
        )

        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            safety_checker=None,
            use_safetensors=True,
            controlnet=controlnet_canny,
            vae=vae,
        )

        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=self.device, dtype=torch_dtype).to(self.device)
        if self.device != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)

        # if args.taesd:
        # self.pipe.vae = AutoencoderTiny.from_pretrained(
        #     taesd_model, torch_dtype=torch_dtype, use_safetensors=True
        # ).to(self.device)

        # if args.torch_compile:
        #     self.pipe.unet = torch.compile(
        #         self.pipe.unet, mode="reduce-overhead", fullgraph=True
        #     )
        #     self.pipe.vae = torch.compile(
        #         self.pipe.vae, mode="reduce-overhead", fullgraph=True
        #     )
        #     self.pipe(
        #         prompt="warmup",
        #         image=[Image.new("RGB", (768, 768))],
        #         control_image=[Image.new("RGB", (768, 768))],
        #     )

    def _initialize_pipeline(self):
        # raise NotImplementedError("Subclasses should implement this method.")
        pass

    def transform(self, image, prompt) -> Image.Image:
        image = self.input_image or image
        negative_prompt = self.negative_prompt
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        # control_image = self.canny_torch(
        #     image, self.canny_low_threshold, self.canny_high_threshold
        # )
        # control_image = get_depth_map(image)
        control_image = get_zoe_depth_map(image).resize((self.width, self.height))
        steps = self.num_steps
        if int(steps * self.strength) < 1:
            steps = math.ceil(1 / max(0.10, self.strength))

        results = self.pipe(
            image=image,
            control_image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=self.generator,
            strength=self.strength,
            num_inference_steps=steps,
            guidance_scale=self.cfg,
            width=self.width,
            height=self.height,
            output_type="pil",
            controlnet_conditioning_scale=self.controlnet_scale,
            control_guidance_start=self.controlnet_start,
            control_guidance_end=self.controlnet_end,
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

    # # Interpolate between the previous frame and the transformed image
    # if previous_frame is not None:
    #     output_frame = interpolate_images(previous_frame, transformed_image, alpha=0.5)
    # else:
    #     output_frame = transformed_image

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


@app.route('/stream')
def stream():
    def generate():
        global current_prompt
        global video_streamer
        global image_transformer
        global previous_frame

        try:
            while True:
                output_frame = transform_frame(
                    previous_frame, 
                    prompt=current_prompt)
                previous_frame = output_frame

                # # set the new image as the previous, to condition on
                # image_transformer.set_image(previous_frame)
                
                img_byte_arr = BytesIO()
                pil_image = convert_to_pil_image(output_frame)
                pil_image.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                yield f"data: {encoded_img}\n\n"
                time.sleep(1 / 30)  # Control frame rate
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
