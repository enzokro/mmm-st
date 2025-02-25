"""
Utilities for the SD 3.5 Video Transformer application.

This module provides utility functions and classes for video processing
and image transformation, including:
- Video capture and streaming
- Image preprocessing for ControlNets:
  - Canny edge detection
  - Depth map generation
  - Gaussian blur
- Image conversion and processing
"""

import logging
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from queue import Queue
import threading
from typing import Optional, Union, Tuple

# Try to import depth models - may not be available in all environments
try:
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    depth_models_available = True
except ImportError:
    depth_models_available = False
    
# Try to import DepthFM for better depth maps
try:
    import depthfm
    from depthfm.dfm import DepthFM
    depthfm_available = True
except ImportError:
    depthfm_available = False

# Import configuration
from config import Config

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize depth estimation models if available
depth_estimator = None
feature_extractor = None
depthfm_model = None

if depth_models_available:
    try:
        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(Config.DEVICE)
        feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        logger.info("Depth estimation models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load depth estimation models: {e}")

if depthfm_available:
    try:
        depthfm_model = DepthFM(ckpt_path="depthfm_model_checkpoints")
        depthfm_model.eval()
        depthfm_model.to(Config.DEVICE)
        logger.info("DepthFM model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load DepthFM model: {e}")


def get_canny_map(
        image: Union[Image.Image, np.ndarray],
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> Image.Image:
    """
    Generate a Canny edge map from an input image.
    
    Args:
        image: Input image (PIL or numpy array)
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
        
    Returns:
        PIL image with Canny edges
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = image
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    try:
        # Apply Canny edge detection
        edges = cv2.Canny(img, low_threshold, high_threshold)
        
        # Convert back to RGB (3 channels)
        edges_rgb = np.stack([edges, edges, edges], axis=2)
        
        return Image.fromarray(edges_rgb)
    except Exception as e:
        logger.error(f"Error generating Canny map: {e}")
        # Return blank image on error
        return Image.new('RGB', image.size if isinstance(image, Image.Image) else (Config.WIDTH, Config.HEIGHT), color='black')


def get_depth_map(image: Image.Image) -> Image.Image:
    """
    Generate a depth map from an input image.
    
    Tries to use DepthFM if available, falls back to DPT model.
    
    Args:
        image: Input PIL image
        
    Returns:
        PIL image representing the depth map
    """
    # Try using DepthFM first (better quality)
    if depthfm_available and depthfm_model is not None:
        try:
            # Convert to tensor
            img_tensor = F.to_tensor(image).unsqueeze(0)
            
            # Get original dimensions
            _, _, h, w = img_tensor.shape
            
            # Resize to 512x512 for inference
            img_tensor = F.resize(img_tensor, (512, 512), interpolation=F.InterpolationMode.BILINEAR)
            
            # Move to device
            img_tensor = img_tensor.to(Config.DEVICE)
            
            # Generate depth map
            with torch.no_grad():
                depth_map = depthfm_model(img_tensor, num_steps=2, ensemble_size=4)
            
            # Resize back to original dimensions
            depth_map = F.resize(depth_map, (h, w), interpolation=F.InterpolationMode.BILINEAR)
            
            # Normalize to 0-1
            depth_min = torch.min(depth_map)
            depth_max = torch.max(depth_map)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
            # Convert to RGB (3 channels) and then to PIL
            depth_map = torch.cat([depth_map] * 3, dim=1)
            depth_map = depth_map.permute(0, 2, 3, 1).cpu().numpy()[0]
            depth_map = (depth_map * 255.0).clip(0, 255).astype(np.uint8)
            
            return Image.fromarray(depth_map)
        except Exception as e:
            logger.warning(f"DepthFM failed, falling back to DPT: {e}")
    
    # Fall back to DPT model
    if depth_models_available and depth_estimator is not None and feature_extractor is not None:
        try:
            # Preprocess image for depth model
            inputs = feature_extractor(images=image, return_tensors="pt").to(Config.DEVICE)
            
            # Generate depth map
            with torch.no_grad(), torch.autocast(Config.DEVICE):
                depth_map = depth_estimator(**inputs).predicted_depth
            
            # Interpolate to target size if needed
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(Config.HEIGHT, Config.WIDTH),
                mode="bicubic",
                align_corners=False,
            )
            
            # Normalize depth values
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            
            # Convert to RGB (3 channels)
            depth_map = torch.cat([depth_map] * 3, dim=1)
            
            # Convert to numpy array and then to PIL
            depth_map = depth_map.permute(0, 2, 3, 1).cpu().numpy()[0]
            depth_map = (depth_map * 255.0).clip(0, 255).astype(np.uint8)
            
            return Image.fromarray(depth_map)
        except Exception as e:
            logger.error(f"Error generating depth map: {e}")
    
    # If all methods fail, return a gray image
    logger.warning("No depth estimation method available, returning gray image")
    return Image.new('RGB', (Config.WIDTH, Config.HEIGHT), color='gray')


def get_blur_map(
        image: Image.Image,
        kernel_size: int = 50
    ) -> Image.Image:
    """
    Generate a blurred version of the input image for Blur ControlNet.
    
    Args:
        image: Input PIL image
        kernel_size: Size of Gaussian blur kernel
        
    Returns:
        Blurred PIL image
    """
    try:
        # Create Gaussian blur transform
        gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size)
        
        # Apply blur
        blurred_image = gaussian_blur(image)
        
        return blurred_image
    except Exception as e:
        logger.error(f"Error generating blur map: {e}")
        # Return original image on error
        return image


class VideoStreamer:
    """
    Class for capturing video frames from a camera or video file.
    
    This class provides methods to initialize a video capture device and
    retrieve frames from it as PIL images.
    """
    
    def __init__(self, device_path=Config.VIDEO_PATH):
        """
        Initialize the video streamer.
        
        Args:
            device_path: Path to the video capture device or index for webcam
        """
        self.cap = cv2.VideoCapture(device_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {device_path}")
            raise ValueError(f"Video source cannot be opened: {device_path}")
            
        # Set capture properties if defined in config
        if hasattr(Config, 'CAP_PROPS') and isinstance(Config.CAP_PROPS, dict):
            for prop, value in Config.CAP_PROPS.items():
                try:
                    prop_id = getattr(cv2, prop)
                    self.cap.set(prop_id, value)
                except AttributeError:
                    logger.warning(f"Unknown OpenCV property: {prop}")
        
        logger.info(f"VideoStreamer initialized with device: {device_path}")

    def get_current_frame(self):
        """
        Get the current frame from the video source.
        
        Returns:
            PIL Image or None if frame cannot be read
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert from BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize if needed
        if (pil_image.width != Config.WIDTH or pil_image.height != Config.HEIGHT) and hasattr(Config, 'WIDTH') and hasattr(Config, 'HEIGHT'):
            pil_image = pil_image.resize((Config.WIDTH, Config.HEIGHT))
            
        return pil_image

    def release(self):
        """Release the video capture resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            logger.info("Video capture released")


def interpolate_images(image1: Image.Image, image2: Image.Image, alpha: float = 0.5) -> Image.Image:
    """
    Interpolate between two images with a given alpha blending factor.
    
    Args:
        image1: First image
        image2: Second image
        alpha: Blending factor (0-1)
        
    Returns:
        Blended image
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    return Image.blend(image1, image2, alpha)


def convert_to_pil_image(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
    """
    Convert various image formats to PIL Image.
    
    Args:
        image: Image to convert (PIL Image, torch.Tensor, numpy.ndarray)
        
    Returns:
        PIL Image
    """
    if isinstance(image, Image.Image):
        return image
    elif torch.is_tensor(image):
        if image.device != torch.device('cpu'):
            image = image.to('cpu')
        image = image.numpy() if image.requires_grad else image.detach().numpy()
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)
        if image.shape[2] == 1:
            image = image[:, :, 0]
        return Image.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float32 else image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return Image.fromarray(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


class SharedQueue:
    """Thread-safe queue for image processing."""
    
    def __init__(self, maxsize=10):
        """
        Initialize the shared queue.
        
        Args:
            maxsize: Maximum queue size
        """
        self.queue = Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        
    def put(self, item):
        """
        Add an item to the queue.
        
        Args:
            item: Item to add
        """
        with self.lock:
            # If queue is full, remove oldest item
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Queue.Empty:
                    pass
            self.queue.put(item)
            
    def get(self):
        """
        Get an item from the queue.
        
        Returns:
            An item from the queue or None if empty
        """
        with self.lock:
            if self.queue.empty():
                return None
            return self.queue.get()
            
    def empty(self):
        """Check if the queue is empty."""
        with self.lock:
            return self.queue.empty()
            
    def full(self):
        """Check if the queue is full."""
        with self.lock:
            return self.queue.full()
            
    def qsize(self):
        """Get the current queue size."""
        with self.lock:
            return self.queue.qsize()


@dataclass
class SharedResources:
    """Thread-safe container for resources shared between threads."""
    frame: Optional[Image.Image] = None
    prompt: Optional[str] = None
    transformer_manager: Optional[Any] = None
    current_model: str = "sd35"
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    connected_clients: int = 0
    frame_params: dict = field(default_factory=dict)
    
    def update_frame(self, frame: Image.Image) -> None:
        """Update the current frame in a thread-safe manner."""
        with self.lock:
            self.frame = frame
    
    def get_frame(self) -> Optional[Image.Image]:
        """Get the current frame in a thread-safe manner."""
        with self.lock:
            return self.frame.copy() if self.frame else None
    
    def update_prompt(self, prompt: str) -> None:
        """Update the current prompt in a thread-safe manner."""
        with self.lock:
            self.prompt = prompt
            logger.info(f"Prompt updated: {prompt}")
    
    def get_prompt(self) -> Optional[str]:
        """Get the current prompt in a thread-safe manner."""
        with self.lock:
            return self.prompt
    
    def set_model(self, model_type: str) -> None:
        """Set the current transformer model type."""
        with self.lock:
            self.current_model = model_type
            if self.transformer_manager:
                self.transformer_manager.set_current_model(model_type)
            logger.info(f"Model type set to: {model_type}")
    
    def get_model(self) -> str:
        """Get the current transformer model type."""
        with self.lock:
            return self.current_model
    
    def update_params(self, params: dict) -> None:
        """Update transformation parameters."""
        with self.lock:
            self.frame_params.update(params)
            logger.info(f"Parameters updated: {params}")
    
    def get_params(self) -> dict:
        """Get current transformation parameters."""
        with self.lock:
            return self.frame_params.copy()
    
    def increment_clients(self) -> int:
        """Increment connected clients count."""
        with self.lock:
            self.connected_clients += 1
            return self.connected_clients
    
    def decrement_clients(self) -> int:
        """Decrement connected clients count."""
        with self.lock:
            self.connected_clients = max(0, self.connected_clients - 1)
            return self.connected_clients


class VideoProcessingThread(threading.Thread):
    """
    Thread for continuous video processing.
    
    This thread continuously captures frames from a video source,
    transforms them based on the current prompt, and updates the
    shared frame buffer.
    """
    
    def __init__(
            self, 
            shared_resources: SharedResources, 
            device_path: str = Config.VIDEO_PATH,
            frame_blend: float = getattr(Config, 'FRAME_BLEND', 0.5)
        ):
        """
        Initialize the video processing thread.
        
        Args:
            shared_resources: Container for resources shared between threads
            device_path: Path to the video capture device
            frame_blend: Blending factor for frame interpolation (0-1)
        """
        super().__init__(daemon=True)
        self.shared_resources = shared_resources
        self.stop_event = shared_resources.stop_event
        self.video_streamer = VideoStreamer(device_path)
        self.previous_frame = None
        self.frame_blend = frame_blend
        logger.info(f"VideoProcessingThread initialized with device: {device_path}")

    def run(self) -> None:
        """Main thread loop for video processing."""
        logger.info("VideoProcessingThread started")
        try:
            while not self.stop_event.is_set():
                # Only process frames if there are connected clients
                if self.shared_resources.connected_clients > 0:
                    # Capture frame
                    current_frame = self.video_streamer.get_current_frame()
                    if current_frame is None:
                        time.sleep(0.01)  # Short sleep to prevent CPU spinning
                        continue
                    
                    # Get current prompt and model
                    current_prompt = self.shared_resources.get_prompt()
                    current_model = self.shared_resources.get_model()
                    
                    # Get current parameters
                    params = self.shared_resources.get_params()
                    
                    # Transform frame if prompt is available and transformer manager exists
                    if current_prompt and self.shared_resources.transformer_manager:
                        try:
                            transformed_frame = self.shared_resources.transformer_manager.transform(
                                current_frame,
                                current_prompt,
                                model_type=current_model,
                                **params
                            )
                        except Exception as e:
                            logger.error(f"Error transforming frame: {str(e)}")
                            transformed_frame = current_frame
                    else:
                        transformed_frame = current_frame
                    
                    # Apply frame interpolation for smoother transitions
                    if self.previous_frame is not None:
                        final_frame = interpolate_images(
                            self.previous_frame, 
                            transformed_frame, 
                            self.frame_blend
                        )
                    else:
                        final_frame = transformed_frame
                    
                    # Update shared frame
                    self.shared_resources.update_frame(final_frame)
                    self.previous_frame = final_frame
                
                # Short sleep to control frame rate
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in VideoProcessingThread: {str(e)}")
        finally:
            self.video_streamer.release()
            logger.info("VideoProcessingThread stopped")