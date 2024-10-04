import gc
import time
import atexit
import base64
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
from flask import Flask, jsonify, request, Response, render_template, stream_with_context
from flask_socketio import SocketIO, emit
from config import Config
from utils import (
    SharedResources, VideoStreamer, interpolate_images, 
    get_depth_map, get_pose_map, convert_to_pil_image, BaseTransformer
)

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins=Config.WEBSOCKET_CORS)

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
        self.cfg = cfg
        self.strength = strength
        self.controlnet_scale = controlnet_scale
        self.controlnet_start = controlnet_start
        self.controlnet_end = controlnet_end
        self.width = width
        self.height = height
        self.generator = torch.Generator(device="cpu").manual_seed(Config.SEED)

    def _initialize_pipeline(self):
        base_model_id = Config.MODEL_NAME
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-lora.safetensors"

        controlnet_depth = ControlNetModel.from_pretrained(
            Config.CONTROLNET_DEPTH,
            torch_dtype=Config.DTYPE
        ).to(self.device)

        controlnet_pose = ControlNetModel.from_pretrained(
            Config.CONTROLNET_POSE,
            torch_dtype=Config.DTYPE
        ).to(self.device)

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=Config.DTYPE
        )

        unet = UNet2DConditionModel.from_config(Config.MODEL_NAME, subfolder="unet").to(self.device, Config.DTYPE)
        unet.load_state_dict(load_file(hf_hub_download(Config.UNET_ID, Config.UNET_CKPT), device=self.device))

        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base_model_id,
            unet=unet,
            controlnet=[controlnet_depth, controlnet_pose],
            vae=vae,
            torch_dtype=Config.DTYPE,
            variant="fp16"
        ).to(self.device)

        self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipe.fuse_lora()

        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

        # final pipeline setup
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device=self.device, dtype=self.dtype).to(self.device)

        if self.device != "mps":
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.controlnet.to(memory_format=torch.channels_last)

        # Initialize latents
        batch_size = 1
        num_channels_latents = self.pipe.unet.config.in_channels
        self.latents_shape = (batch_size, num_channels_latents, self.height // self.pipe.vae_scale_factor, self.width // self.pipe.vae_scale_factor)
        self.latents = self.refresh_latents()

    def transform(self, image, prompt):
        depth_image = get_depth_map(image)
        pose_image = get_pose_map(image)

        results = self.pipe(
            prompt=prompt,
            image=image,
            control_image=[depth_image, pose_image],
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_steps,
            guidance_scale=self.cfg,
            strength=self.strength,
            controlnet_conditioning_scale=self.controlnet_scale,
            control_guidance_start=self.controlnet_start,
            control_guidance_end=self.controlnet_end,
            width=self.width,
            height=self.height,
            generator=self.generator,
            latents=self.get_latents(),
            output_type="pil"
        )

        return results.images[0]

    def get_latents(self):
        return self.latents.clone()

    def refresh_latents(self):
        self.latents = torch.randn(self.latents_shape, generator=self.generator, device=self.device, dtype=Config.DTYPE)
        return self.latents

# Initialize shared resources
shared_resources = SharedResources()
shared_resources.image_transformer = SDXL_Hyper(
    num_steps=Config.NUM_STEPS,
    img_size=(Config.WIDTH, Config.HEIGHT)
)

# Video processing thread
class VideoProcessingThread(threading.Thread):
    def __init__(self, shared_resources, device_path=Config.VIDEO_PATH):
        super().__init__(daemon=True)
        self.shared_resources = shared_resources
        self.stop_event = shared_resources.stop_event
        self.video_streamer = VideoStreamer(device_path)
        self.previous_frame = None

    def run(self):
        while not self.stop_event.is_set():
            current_frame = self.video_streamer.get_current_frame()
            if current_frame is None:
                continue
            
            current_prompt = self.shared_resources.get_prompt()

            if current_prompt:
                transformed_frame = self.shared_resources.image_transformer.transform(
                    current_frame,
                    current_prompt,
                )
            else:
                transformed_frame = current_frame

            if self.previous_frame is not None:
                final_frame = interpolate_images(self.previous_frame, transformed_frame, Config.FRAME_BLEND)
            else:
                final_frame = transformed_frame

            self.shared_resources.update_frame(final_frame)
            self.previous_frame = final_frame

        self.video_streamer.release()

# Start video processing thread
video_thread = VideoProcessingThread(shared_resources)
video_thread.start()

@app.route('/')
def index():
    return render_template('index_socket.html')

@app.route('/set_prompt', methods=['POST'])
def set_prompt():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    shared_resources.update_prompt(prompt)
    return jsonify({"message": "Prompt set successfully"})

@app.route('/refresh_latents', methods=['POST'])
def refresh_latents():
    try:
        signal = request.json.get('signal')
        if signal != "refresh":
            return jsonify({"error": "Invalid signal"}), 400
        
        with shared_resources.lock:
            shared_resources.image_transformer.refresh_latents()
        
        return jsonify({"message": "Latents refreshed"})
    except Exception as e:
        return jsonify({"error": f"Could not refresh latents: {str(e)}"}), 500

@socketio.on('connect')
def handle_connect():
    emit('connected', {'message': 'Connection established'})

@socketio.on('stream')
def stream():
    while not shared_resources.stop_event.is_set():
        frame = shared_resources.get_frame()
        if frame:
            img_byte_arr = BytesIO()
            frame = convert_to_pil_image(frame)
            frame = frame.resize((Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
            frame.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            emit('frame', {'data': encoded_img})
        time.sleep(0.1)

def cleanup():
    shared_resources.stop_event.set()
    video_thread.join()
    gc.collect()
    torch.cuda.empty_cache()

atexit.register(cleanup)

if __name__ == "__main__":
    socketio.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        debug=True, 
        threaded=True, 
        use_reloader=False,
    )