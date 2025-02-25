"""
Configuration for the SD 3.5 Video Transformer application.

This module defines various configuration parameters for the application,
including model settings, video dimensions, server settings, etc.
"""

import torch

# Determine available device with appropriate settings
if torch.cuda.is_available():
    device = "cuda"
    # Optimize CUDA performance
    torch._dynamo.config.verbose = True
    torch._dynamo.config.suppress_errors = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = "mps"  # For Apple Silicon
else:
    device = "cpu"

class Config:
    """Configuration parameters for the application."""
    
    # Basic app parameters
    HOST = '0.0.0.0'
    PORT = 8989
    SEED = 12297829382473034410  # Random seed for reproducibility
    
    # Video settings
    VIDEO_PATH = 0  # Webcam index (0 for default camera)
    CAP_PROPS = {'CAP_PROP_FPS': 30}  # OpenCV capture properties
    FRAME_BLEND = 0.7  # Blending factor for frame interpolation (higher = smoother but more latency)
    FRAME_SKIP = 2  # Only send every nth frame to reduce bandwidth
    
    # Image dimensions
    WIDTH = 768  # Width for model input/output
    HEIGHT = 768  # Height for model input/output
    DISPLAY_WIDTH = 640  # Width for web display
    DISPLAY_HEIGHT = 640  # Height for web display
    
    # Model parameters
    DEVICE = device
    DTYPE = torch.float16 if device != "cpu" else torch.float32
    
    # Model paths
    SD35_TURBO_MODEL = "stabilityai/stable-diffusion-3.5-large-turbo"
    SD35_BASE_MODEL = "stabilityai/stable-diffusion-3.5-large"  # Used for ControlNets
    
    # ControlNet paths
    CONTROLNET_CANNY = "stabilityai/control-net-canny-sd3.5-large"
    CONTROLNET_DEPTH = "stabilityai/control-net-depth-sd3.5-large"
    CONTROLNET_BLUR = "stabilityai/control-net-blur-sd3.5-large"
    
    # Default generation parameters
    NUM_STEPS = 4  # Default for Turbo model
    CONTROLNET_STEPS = 50  # Recommended higher value for ControlNets
    CFG = 0.0  # Default guidance scale for Turbo
    CONTROLNET_CFG = 5.0  # Non-zero guidance typically better for ControlNets
    STRENGTH = 0.75  # How much to preserve from original image (lower = more creative)
    CONTROLNET_SCALE = 0.75  # Weight of ControlNet conditioning
    NEGATIVE_PROMPT = "low quality, blurry, distorted, ugly, pixelated"  # Default negative prompt
    
    # Preprocessing parameters
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 200
    BLUR_KERNEL_SIZE = 50
    
    # WebSocket settings
    WEBSOCKET_CORS = "*"
    
    # Model quantization
    USE_QUANTIZATION = True  # Whether to use 4-bit quantization for lower VRAM usage
    
    # Available model types
    MODEL_TYPES = ['sd35', 'canny', 'depth', 'blur']
    DEFAULT_MODEL = 'sd35'
    
    # UI layout
    UI_COLUMNS = 2  # Number of columns for the UI layout