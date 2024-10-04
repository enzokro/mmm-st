import logging
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from controlnet_aux import OpenposeDetector
from config import Config
from queue import Queue
import threading

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Depth map utils
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

# Pose estimation
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(Config.DEVICE)

def get_pose_map(image):
    pose_image = openpose(image)
    pose_image = pose_image.resize((Config.WIDTH, Config.HEIGHT))
    return pose_image

class VideoStreamer:
    def __init__(self, device_path=Config.VIDEO_PATH):
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
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    return Image.blend(image1, image2, alpha)

def convert_to_pil_image(image):
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
        raise TypeError("Unsupported image type")

class SharedResources:
    def __init__(self):
        self.current_frame = None
        self.current_prompt = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame_queue = Queue()

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame
            self.frame_queue.put(frame)

    def get_frame(self):
        return self.frame_queue.get()

    def update_prompt(self, prompt):
        with self.lock:
            self.current_prompt = prompt

    def get_prompt(self):
        with self.lock:
            return self.current_prompt

# Base transformer (moved from previous implementation)
class BaseTransformer:
    def __init__(
            self,
            model_name=Config.MODEL_NAME,
            device=Config.DEVICE,
            img_size=(Config.WIDTH, Config.HEIGHT),
            num_steps=Config.NUM_STEPS,
            negative_prompt=Config.NEGATIVE_PROMPT, 
        ):
        self.model_name = model_name
        self.device = device
        self.img_size = img_size
        self.num_steps = num_steps
        self.negative_prompt = negative_prompt
        self.input_image = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, image, prompt):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def prepare_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.resize(self.img_size)
    
    def set_image(self, image):
        self.input_image = image
    
    def __call__(self, image, prompt, **kwargs):
        return self.transform(image, prompt)