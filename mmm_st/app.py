import gc
import math
import time
import base64
import atexit
import logging
import threading
from io import BytesIO
from queue import Queue
from PIL import Image
import numpy as np
import torch
import torch._dynamo as dynamo
import cv2
from fastcore.basics import store_attr
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers import EulerDiscreteScheduler, DDIMScheduler, DDPMScheduler, KDPM2DiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import DPTImageProcessor, DPTForDepthEstimation
from controlnet_aux import OpenposeDetector
from huggingface_hub import hf_hub_download
from flask import Flask, jsonify, request, Response, render_template, stream_with_context
from flask_socketio import SocketIO, emit
from mmm_st.config import Config as BaseConfig
from mmm_st.diffuse import BaseTransformer
from mmm_st.video import convert_to_pil_image



# Create Flask app
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# set the proper device
if torch.cuda.is_available():
    device = "cuda" #torch.device("cuda")
    # test optimizations
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = "mps" # torch.device("mps")
else:
    device = "cpu" #torch.device("cpu")


# parameters for the run
class Config:

    # basic app params
    HOST = '0.0.0.0'
    PORT = 8989
    SEED  = BaseConfig.SEED
    VIDEO_PATH = '/dev/video0'
    CAP_PROPS = {}

    # for the denoised image
    NUM_STEPS = 4
    HEIGHT = 1024
    WIDTH = 1024

    # size of the final image sent over
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080

    # alpha blending factor for frame interpolation
    FRAME_BLEND = 0.75

    # default data type for torch
    DTYPE = torch.float16

    # models for denoising, and controlnet
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    UNET_ID = "ByteDance/SDXL-Lightning"
    UNET_CKPT = "sdxl_lightning_4step_unet.safetensors"
    CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
    CONTROLNET_POSE = "thibaud/controlnet-openpose-sdxl-1.0"
    
    # scheduler type
    SCHEDULER = 'ddim'


# fix the seed for reproducibility
torch.manual_seed(Config.SEED)

# map from name to scheduler 
name2sched = {
    'ddpm':  DDPMScheduler,
    'ddim':  DDIMScheduler,
    'euler': EulerDiscreteScheduler,
    'ddpm2': KDPM2DiscreteScheduler,
}


class SharedResources:
    """Manages shared resources.
    
    Creates the following:
        - Stable Diffusion pipeline
        - Current input prompt
        - Current generated frame

    Uses locks, events, and queues for concurrency.
    """
    def __init__(self):
        self.image_transformer = SDXL(
            num_steps=Config.NUM_STEPS,
            height=Config.HEIGHT,
            width=Config.WIDTH,
        )
        self.current_frame = None
        self.current_prompt = None
        self.lock = threading.Lock() 
        self.stop_event = threading.Event() 
        self.frame_queue = Queue() 

    def update_frame(self, frame):
        "Places a new `frame` in the queue."
        with self.lock:
            self.current_frame = frame
            self.frame_queue.put(frame) 

    def get_frame(self):
        return self.frame_queue.get()

    def update_prompt(self, prompt):
        "Updates the generation prompt."
        with self.lock:
            self.current_prompt = prompt

    def get_prompt(self):
        "Gets the current generation prompt."
        with self.lock:
            return self.current_prompt


class VideoStreamer:
    "Reads frames from a video capture source."
    def __init__(self, device_path: str):
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


def interpolate_images(image1, image2, alpha=0.5):
    """Interpolate two images with a given `alpha` blending factor.
    
    output = (1 - alpha) * image1 + (alpha * image2)
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    return Image.blend(image1, image2, alpha)
       
       
# Video processing class that handles the transformation and streaming
class VideoProcessingThread(threading.Thread):
    "Background thread for video processing."
    def __init__(self, shared_resources, device_path=Config.VIDEO_PATH):
        super().__init__(daemon=True)
        self.shared_resources = shared_resources
        self.stop_event = shared_resources.stop_event

        # create the video streamer
        self.video_streamer = VideoStreamer(device_path)
        self.previous_frame = None

    def run(self):
        "Continuously transforms input video frames."
        while not self.stop_event.is_set():

            # grab the current frame
            current_frame = self.video_streamer.get_current_frame()
            if current_frame is None:
                continue
            
            # grab the current prompt
            current_prompt = self.shared_resources.get_prompt()

            # transform the current frame
            if current_prompt not in (None, ""):
                transformed_frame = self.shared_resources.image_transformer.transform(
                    current_frame,
                    current_prompt,
                )
            else:
                transformed_frame = current_frame

            # interpolate it with the previous frame 
            if self.previous_frame is not None:
                final_frame = interpolate_images(self.previous_frame, transformed_frame, Config.FRAME_BLEND)
            else:
                final_frame = transformed_frame

            # update the final frames
            self.shared_resources.update_frame(final_frame)
            self.previous_frame = final_frame

            # sanity check for stopping
            if self.stop_event.is_set(): break

        # at the end, release the video source
        self.video_streamer.release()


class SDXL(BaseTransformer):
    """Uses an SDXL model with ControlNet to generate images.
    
    The pipeline uses the 4-step UNet from SDXL-Lightning for faster generations.
    It uses two ControlNet modules:
        - One for depth
        - Another for pose
    """
    def __init__(
            self,
            cfg=1.0,
            strength=0.75,
            canny_low_threshold=100,
            canny_high_threshold=200,
            controlnet_scale=[0.5, 0.5],
            controlnet_start=0.0,
            controlnet_end=1.0,
            width=Config.WIDTH,
            height=Config.HEIGHT,
            dtype=Config.DTYPE,
            scheduler=Config.SCHEDULER,
            device=device,
            **kwargs,
        ):
        store_attr()
        self.generator = torch.Generator(device="cpu").manual_seed(Config.SEED)
        super().__init__(device=device, **kwargs)

    def _initialize_pipeline(self):
        "Sets up the pipeline."

        # initialize the depth control net
        controlnet_depth = ControlNetModel.from_pretrained(
            Config.CONTROLNET_DEPTH,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)

        # initialize the pose control net
        controlnet_pose = ControlNetModel.from_pretrained(
            Config.CONTROLNET_POSE,
            torch_dtype=self.dtype,
            # use_safetensors=True,
        ).to(self.device) 

        # model for depth estimation
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
        self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

        # model for pose estimation
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(self.device)

        # load the Lightning UNet
        unet_config = UNet2DConditionModel.load_config(Config.MODEL_ID, subfolder="unet")
        unet = UNet2DConditionModel.from_config(unet_config).to(self.device, self.dtype)
        unet.load_state_dict(
            load_file(hf_hub_download(Config.UNET_ID, Config.UNET_CKPT), device=self.device)
        )

        # load in the fixed VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=self.dtype
        )
        
        # load the model with controlnets.
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            Config.MODEL_ID,
            use_safetensors=True,
            unet=unet,
            vae=vae,
            # NOTE: try using the same contorlnet twice, with different strengths and images
            controlnet=[controlnet_depth, controlnet_pose],
            # controlnet=controlnet_depth,
        )

        # sampler with trailing timesteps per the docs
        # self.pipe.scheduler = EulerDiscreteScheduler.from_config(
        self.pipe.scheduler = name2sched[self.scheduler].from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
        )

        # final pipeline setup
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=self.device, dtype=self.dtype).to(self.device)
        # optimizations assuming we're on a ~new GPU
        if self.device != torch.device("mps"):
            print("Set UNet to channels_last")
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.controlnet.to(memory_format=torch.channels_last)
        # print("Compiling the UNets...")
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        # self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead", fullgraph=True)
        # print("UNet compiled.")

        # start with a fixed set of latents
        batch_size, num_images_per_prompt = 1, 1
        batch_size *= num_images_per_prompt
        num_channels_latents = self.pipe.unet.config.in_channels
        self.latents_shape = (batch_size, num_channels_latents, self.height // self.pipe.vae_scale_factor, self.width // self.pipe.vae_scale_factor)
        self.latents = self.refresh_latents()

    def transform(self, image, prompt) -> Image.Image:
        """Transforms the given `image` based on the `prompt`.
        
        Uses a controlnet to condition and manage the generation. 
        """

        # get the controlnet depth image
        depth_image = self.get_depth_map(image)

        # get the controlnet pose image
        pose_image = self.get_pose_map(image)

        # compute the number of steps
        steps = self.num_steps
        if int(steps * self.strength) < 1:
            steps = math.ceil(1 / max(0.10, self.strength))

        # generate and return the image
        results = self.pipe(
            prompt=prompt,
            image=[depth_image, pose_image],
            # image=depth_image,
            negative_prompt=self.negative_prompt,
            controlnet_conditioning_scale=self.controlnet_scale,
            num_inference_steps=steps,
            guidance_scale=self.cfg,
            strength=self.strength,
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
        "Builds the controlnet depth map."
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
    
    def get_pose_map(self, image):
        "Builds the controlnet pose map."
        # with torch.no_grad(), torch.autocast(self.device):
        pose_image = self.openpose(image)
        pose_image = pose_image.resize((self.width, self.height))
        return pose_image
    
    def get_latents(self):
        "Returns a copy of the current latents, to prevent them from being overriden."
        return self.latents.clone()

    def refresh_latents(self):
        "Creates a fixed set of latents for ~reproducible images."
        latents = randn_tensor(self.latents_shape, generator=self.generator, device=torch.device(self.device), dtype=self.dtype)
        return latents
    

# create the shared resources and start the video processing thread
shared_resources = SharedResources()
video_thread = VideoProcessingThread(shared_resources, device_path=Config.VIDEO_PATH)
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
        
        # refresh the latents
        with shared_resources.lock:
            shared_resources.image_transformer.refresh_latents()
        
        return jsonify({"message": "Latents refreshed"})
    except Exception as e:
        logger.error(f"Error refreshing latents: {e}")
        return jsonify({"error": "Could not refresh latents"}), 500


# # Flask endpoint to stream the latest frame
# @app.route('/stream')
# def stream():
#     def generate():
#         while not shared_resources.stop_event.is_set():
#             frame = shared_resources.get_frame()
#             if frame is not None:
#                 # rescale to better fit the image
#                 frame = convert_to_pil_image(frame)
#                 frame = frame.resize((Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
#                 img_byte_arr = BytesIO()
#                 frame.save(img_byte_arr, format='JPEG')
#                 img_byte_arr.seek(0)
#                 encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
#                 yield f"data: {encoded_img}\n\n"
#                 time.sleep(1 / 30)  # Frame rate control

#     return Response(stream_with_context(generate()), mimetype='text/event-stream')


# WebSocket events for connection and streaming
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'message': 'Connection established'})

@socketio.on('stream')
def stream():
    while not shared_resources.stop_event.is_set():
        frame = shared_resources.get_frame()
        if frame:
            img_byte_arr = BytesIO()
            # rescale to better display the image
            frame = convert_to_pil_image(frame)
            frame = frame.resize((Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
            frame.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            emit('frame', {'data': encoded_img})  # Send frame to connected clients
        time.sleep(0.1)  # Control frame rate


def cleanup():
    "Cleanup resources at the end."
    shared_resources.stop_event.set() 
    video_thread.join() 
    gc.collect()  
    torch.cuda.empty_cache()  

# leave it better than we found it
atexit.register(cleanup)

if __name__ == "__main__":
    app.run(
        host=Config.HOST, 
        port=Config.PORT,
        debug=True, 
        threaded=True, 
        use_reloader=False,
    )
