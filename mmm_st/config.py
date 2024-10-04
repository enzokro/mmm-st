import torch

# Device selection
if torch.cuda.is_available():
    device = "cuda"
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

class Config:
    # Basic app params
    HOST = '0.0.0.0'
    PORT = 8989
    SEED = 12297829382473034410
    VIDEO_PATH = 1  # '/dev/video0' # Can be int for camera index or str for file path
    CAP_PROPS = {'CAP_PROP_FPS': 30}

    # Image dimensions
    HEIGHT = 1024
    WIDTH = 1024
    DISPLAY_WIDTH = 1920
    DISPLAY_HEIGHT = 1080

    # Model parameters
    DEVICE = device
    DTYPE = torch.float16
    MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
    CONTROLNET_POSE = "thibaud/controlnet-openpose-sdxl-1.0"

    # Hyper-SD specific
    UNET_ID = "ByteDance/SDXL-Lightning"
    UNET_CKPT = "sdxl_lightning_4step_unet.safetensors"

    # Generation parameters
    NUM_STEPS = 4
    CFG = 0.0
    STRENGTH = 0.75
    CONTROLNET_SCALE = [0.5, 0.5]  # For depth and pose
    CONTROLNET_START = 0.0
    CONTROLNET_END = 1.0
    NEGATIVE_PROMPT = "low quality, blurry, distorted"

    # Frame processing
    FRAME_BLEND = 0.75

    # Scheduler
    SCHEDULER = 'ddim'

    # Websocket
    WEBSOCKET_CORS = "*"