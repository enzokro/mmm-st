# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_config.ipynb.

# %% auto 0
__all__ = ['config', 'Config']

# %% ../nbs/03_config.ipynb 2
import json
import os

# %% ../nbs/03_config.ipynb 3
config = {
  "tfm_type": "regular",
  "models": {
    # "image_to_image": "kandinsky-community/kandinsky-2-2-decoder",
    "image_to_image": "runwayml/stable-diffusion-v1-5",
    "control_net_canny": "lllyasviel/sd-controlnet-canny",
    "control_net_pose": "fusing/stable-diffusion-v1-5-controlnet-openpose",
    "stable_diffusion": "runwayml/stable-diffusion-v1-5",
    "pose_det_model": "lllyasviel/ControlNet",
  },
  "app_settings": {
    "threshold": 0.6,
    "num_valid_frames": 15,
    "targets": ["person", "cat"],
    "host": "localhost",
    "port": 8989
  },
  "image_size": (256, 256),
  "num_steps": 20,
  "device": "cuda",
  "seed": 12297829382473034410,
  "cap_props": {
    "CAP_PROP_FPS": 30
  },
  "negative_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
}


class Config:
    """
    Configuration for the application, loaded from a JSON file.
    """
    # # Load configuration from a JSON file
    # with open('config.json', 'r') as config_file:
    #     config = json.load(config_file)

    # Models configuration
    TRANSFORM_TYPE = config['tfm_type']
    MODEL_NAME = config['models']['image_to_image']
    CONTROL_NET_CANNY_MODEL = config['models']['control_net_canny']
    CONTROL_NET_POSE_MODEL = config['models']['control_net_pose']
    STABLE_DIFFUSION_MODEL = config['models']['stable_diffusion']
    POSE_DET_MODEL = config['models']['pose_det_model']
    DEVICE = config['device']
    SEED = config['seed']

    # Image size
    IMAGE_SIZE = config['image_size']
    NUM_STEPS = config['num_steps']
    NEGATIVE_PROMPT = config['negative_prompt']


    # Application settings
    HOST = os.environ.get('HOST', config['app_settings']['host'])
    PORT = int(os.environ.get('PORT', config['app_settings']['port']))
    DETECTION_URL = f'http://{HOST}:{PORT}/detections'

    # Camera stream properties
    CAP_PROPS = config['cap_props']
