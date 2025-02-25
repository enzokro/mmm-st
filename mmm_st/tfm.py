"""
Transformer models for image generation using Stable Diffusion 3.5 and ControlNets.

This module provides classes for different image transformation models:
- SD35Transformer: Uses Stable Diffusion 3.5 Large Turbo for fast, high-quality generation
- ControlNetTransformer: Base class for ControlNet implementations
- CannyControlTransformer: Uses Canny edge detection for structure guidance
- DepthControlTransformer: Uses depth maps for 3D-aware generation
- BlurControlTransformer: Uses blurred images for high-fidelity upscaling
"""

import os
import gc
import time
import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod

# Import diffusers components
from diffusers import (
    StableDiffusion3Pipeline,
    ControlNetModel,
    StableDiffusion3ControlNetPipeline,
    BitsAndBytesConfig,
)
from transformers import T5EncoderModel
from diffusers.models import SD3Transformer2DModel
from huggingface_hub import hf_hub_download

# Import utility functions
from utils import convert_to_pil_image, get_depth_map, get_canny_map, get_blur_map

# Import configuration
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
if hasattr(Config, 'SEED'):
    torch.manual_seed(Config.SEED)


class BaseTransformer(ABC):
    """
    Abstract base class for image transformers.
    
    This class provides the common interface and functionality for all transformer
    implementations.
    """
    
    def __init__(
            self,
            model_name: str,
            device: str = Config.DEVICE,
            dtype: torch.dtype = getattr(Config, 'DTYPE', torch.float16),
            img_size: Tuple[int, int] = (Config.WIDTH, Config.HEIGHT),
            negative_prompt: Optional[str] = None,
            use_quantization: bool = getattr(Config, 'USE_QUANTIZATION', False),
            **kwargs
        ):
        """
        Initialize the base transformer.
        
        Args:
            model_name: Name or path of the model
            device: Device to run the model on ('cuda', 'mps', 'cpu')
            dtype: Data type for model computation
            img_size: Size of input/output images (width, height)
            negative_prompt: Negative prompt to guide generation away from certain concepts
            use_quantization: Whether to use 4-bit quantization (helps with VRAM usage)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.img_size = img_size
        self.negative_prompt = negative_prompt
        self.use_quantization = use_quantization
        self.kwargs = kwargs
        self.pipe = None
        
        # Initialize the model (will be implemented by subclasses)
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}, device: {device}")
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and initialize the pipeline."""
        pass
    
    @abstractmethod
    def transform(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Transform the input image based on the prompt."""
        pass
    
    def apply_memory_optimizations(self) -> None:
        """Apply memory optimizations for better performance on GPU."""
        if self.pipe is None:
            logger.warning("Pipeline not initialized, skipping memory optimizations")
            return
            
        try:
            # Enable attention slicing for lower memory usage
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing")
            
            # Enable model CPU offload if available and if not using quantization
            # (quantization and offload together can cause issues)
            if not self.use_quantization and hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
                logger.info("Enabled model CPU offload")
                
            # Enable flash attention if using CUDA and function exists
            if self.device == "cuda" and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {str(e)}")
                    
            # Use channels_last memory format for better performance on CUDA
            if self.device == "cuda" and hasattr(self.pipe, "unet"):
                self.pipe.unet.to(memory_format=torch.channels_last)
                logger.info("Set channels_last memory format")
                
            logger.info("Memory optimizations applied")
        except Exception as e:
            logger.warning(f"Error applying memory optimizations: {str(e)}")
    
    def prepare_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for processing."""
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Resize to target size
        if image.size != self.img_size:
            image = image.resize(self.img_size)
            
        return image
    
    def __call__(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """Convenience method to call transform."""
        return self.transform(image, prompt, **kwargs)
        
    def cleanup(self) -> None:
        """Clean up resources to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"{self.__class__.__name__} resources cleaned up")


class SD35Transformer(BaseTransformer):
    """
    Transformer using Stable Diffusion 3.5 Large Turbo.
    
    This class implements image transformation using the latest SD 3.5 Large Turbo
    model, which is optimized for high quality with few inference steps.
    """
    
    def __init__(
            self,
            model_name: str = "stabilityai/stable-diffusion-3.5-large-turbo",
            num_inference_steps: int = 4,
            guidance_scale: float = 0.0,
            strength: float = 0.75,
            max_sequence_length: int = 77,
            use_quantization: bool = False,
            **kwargs
        ):
        """
        Initialize SD35Transformer.
        
        Args:
            model_name: Model name or path
            num_inference_steps: Number of denoising steps (4 is default for Turbo)
            guidance_scale: Classifier-free guidance scale (0.0 is default for Turbo)
            strength: Strength of conditioning (for img2img)
            max_sequence_length: Maximum text prompt length
            use_quantization: Whether to use 4-bit quantization
            **kwargs: Additional parameters for BaseTransformer
        """
        super().__init__(model_name=model_name, use_quantization=use_quantization, **kwargs)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.max_sequence_length = max_sequence_length
        
        # Load the model
        self.load_model()
        
    def load_model(self) -> None:
        """Load the SD 3.5 Large Turbo model."""
        try:
            # Get HF token from environment or config
            hf_token = os.environ.get("HF_TOKEN") or getattr(Config, "HF_TOKEN", None)
            if not hf_token:
                logger.warning("No HF_TOKEN found. Some models may require authentication.")
                
            if self.use_quantization:
                # Set up quantization config for 4-bit loading
                logger.info("Loading model with 4-bit quantization")
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.dtype
                )
                
                # Load the transformer with quantization
                transformer = SD3Transformer2DModel.from_pretrained(
                    self.model_name,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=self.dtype,
                    token=hf_token
                )
                
                # Load quantized T5 (if needed)
                try:
                    t5_nf4 = T5EncoderModel.from_pretrained(
                        "diffusers/t5-nf4", 
                        torch_dtype=self.dtype
                    )
                    
                    # Create pipeline with quantized components
                    self.pipe = StableDiffusion3Pipeline.from_pretrained(
                        self.model_name,
                        transformer=transformer,
                        text_encoder_3=t5_nf4,
                        torch_dtype=self.dtype,
                        token=hf_token
                    )
                except Exception as e:
                    logger.warning(f"Failed to load quantized T5: {e}. Falling back to full pipeline with quantized transformer.")
                    self.pipe = StableDiffusion3Pipeline.from_pretrained(
                        self.model_name,
                        transformer=transformer,
                        torch_dtype=self.dtype,
                        token=hf_token
                    )
            else:
                # Load the full model at target precision
                logger.info(f"Loading model in {self.dtype} precision")
                self.pipe = StableDiffusion3Pipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    token=hf_token
                )
            
            # Move to device and apply optimizations
            self.pipe.to(self.device)
            self.apply_memory_optimizations()
            
            # Set progress bar config
            self.pipe.set_progress_bar_config(disable=True)
            
            logger.info(f"SD35Transformer initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading SD35 model: {str(e)}")
            raise RuntimeError(f"Failed to initialize SD35 pipeline: {str(e)}")
    
    def transform(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """
        Transform the input image using SD 3.5 Large Turbo.
        
        Args:
            image: Input image
            prompt: Text prompt to guide generation
            **kwargs: Additional parameters to override defaults
            
        Returns:
            Transformed image
        """
        try:
            # Prepare image
            image = self.prepare_image(image)
            
            # Set up generation parameters
            inference_steps = kwargs.get('num_inference_steps', self.num_inference_steps)
            guidance_scale = kwargs.get('guidance_scale', self.guidance_scale)
            strength = kwargs.get('strength', self.strength)
            negative_prompt = kwargs.get('negative_prompt', self.negative_prompt)
            max_sequence_length = kwargs.get('max_sequence_length', self.max_sequence_length)
            
            # Get generator if provided
            generator = kwargs.get('generator', None)
            
            # Run the pipeline
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    image=image,  # SD3 supports img2img directly
                    negative_prompt=negative_prompt,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                    max_sequence_length=max_sequence_length,
                    output_type="pil"
                )
            
            # Return the generated image
            return result.images[0]
        except Exception as e:
            logger.error(f"Error in SD35 transform: {str(e)}")
            # Return original image on error
            return image


class ControlNetTransformer(BaseTransformer):
    """
    Base class for ControlNet transformers.
    
    This class provides common functionality for all ControlNet implementations.
    Subclasses should implement the specific control type and preprocessing.
    """
    
    def __init__(
            self,
            model_name: str = "stabilityai/stable-diffusion-3.5-large",
            controlnet_name: str = None,
            controlnet_scale: float = 0.75,
            num_inference_steps: int = 50,  # Higher value (50-60) recommended for ControlNets
            guidance_scale: float = 5.0,    # Non-zero guidance typically better for ControlNets
            use_quantization: bool = False,
            **kwargs
        ):
        """
        Initialize ControlNetTransformer.
        
        Args:
            model_name: Base model name
            controlnet_name: ControlNet model name
            controlnet_scale: Weight of ControlNet conditioning
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            use_quantization: Whether to use quantization
            **kwargs: Additional parameters for BaseTransformer
        """
        super().__init__(model_name=model_name, use_quantization=use_quantization, **kwargs)
        self.controlnet_name = controlnet_name
        self.controlnet_scale = controlnet_scale
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.controlnet = None
        
        # Load the model
        self.load_model()
    
    def load_model(self) -> None:
        """Load the SD 3.5 model with ControlNet."""
        try:
            # Get HF token from environment or config
            hf_token = os.environ.get("HF_TOKEN") or getattr(Config, "HF_TOKEN", None)
            if not hf_token:
                logger.warning("No HF_TOKEN found. Some models may require authentication.")
            
            # First load the controlnet model
            logger.info(f"Loading ControlNet: {self.controlnet_name}")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_name,
                torch_dtype=self.dtype,
                token=hf_token
            )
            
            # Load the pipeline with the controlnet
            if self.use_quantization:
                # Quantization for ControlNet not fully implemented in this version
                # This is a placeholder for future implementation
                logger.warning("Quantization for ControlNet not fully implemented. Using full precision.")
            
            logger.info(f"Loading SD 3.5 with ControlNet: {self.model_name}")
            self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
                self.model_name,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                token=hf_token
            )
            
            # Move to device and apply optimizations
            self.pipe.to(self.device)
            self.apply_memory_optimizations()
            
            # Set progress bar config
            self.pipe.set_progress_bar_config(disable=True)
            
            logger.info(f"ControlNetTransformer initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading ControlNet model: {str(e)}")
            raise RuntimeError(f"Failed to initialize ControlNet pipeline: {str(e)}")
    
    @abstractmethod
    def preprocess_control_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for ControlNet conditioning."""
        pass
    
    def transform(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        """
        Transform image with ControlNet guidance.
        
        Args:
            image: Input image
            prompt: Text prompt
            **kwargs: Additional parameters
            
        Returns:
            Transformed image
        """
        try:
            # Prepare input image
            image = self.prepare_image(image)
            
            # Preprocess for ControlNet (implemented by subclasses)
            control_image = self.preprocess_control_image(image)
            
            # Set up generation parameters
            inference_steps = kwargs.get('num_inference_steps', self.num_inference_steps)
            guidance_scale = kwargs.get('guidance_scale', self.guidance_scale)
            controlnet_scale = kwargs.get('controlnet_scale', self.controlnet_scale)
            negative_prompt = kwargs.get('negative_prompt', self.negative_prompt)
            generator = kwargs.get('generator', None)
            
            # Run the pipeline
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    image=image,
                    control_image=control_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_scale,
                    generator=generator,
                    output_type="pil"
                )
            
            return result.images[0]
        except Exception as e:
            logger.error(f"Error in ControlNet transform: {str(e)}")
            # Return original image on error
            return image


class CannyControlTransformer(ControlNetTransformer):
    """ControlNet transformer using Canny edge detection."""
    
    def __init__(
            self,
            controlnet_name: str = "stabilityai/control-net-canny-sd3.5-large",
            canny_low_threshold: int = 100,
            canny_high_threshold: int = 200,
            **kwargs
        ):
        """
        Initialize CannyControlTransformer.
        
        Args:
            controlnet_name: Canny ControlNet model name
            canny_low_threshold: Lower threshold for Canny edge detection
            canny_high_threshold: Upper threshold for Canny edge detection
            **kwargs: Additional parameters for ControlNetTransformer
        """
        super().__init__(controlnet_name=controlnet_name, **kwargs)
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
    
    def preprocess_control_image(self, image: Image.Image) -> Image.Image:
        """Generate Canny edge map from image."""
        return get_canny_map(
            image, 
            low_threshold=self.canny_low_threshold,
            high_threshold=self.canny_high_threshold
        )


class DepthControlTransformer(ControlNetTransformer):
    """ControlNet transformer using depth maps."""
    
    def __init__(
            self,
            controlnet_name: str = "stabilityai/control-net-depth-sd3.5-large",
            **kwargs
        ):
        """
        Initialize DepthControlTransformer.
        
        Args:
            controlnet_name: Depth ControlNet model name
            **kwargs: Additional parameters for ControlNetTransformer
        """
        super().__init__(controlnet_name=controlnet_name, **kwargs)
    
    def preprocess_control_image(self, image: Image.Image) -> Image.Image:
        """Generate depth map from image."""
        return get_depth_map(image)


class BlurControlTransformer(ControlNetTransformer):
    """ControlNet transformer using blurred images for upscaling."""
    
    def __init__(
            self,
            controlnet_name: str = "stabilityai/control-net-blur-sd3.5-large",
            blur_kernel_size: int = 50,
            **kwargs
        ):
        """
        Initialize BlurControlTransformer.
        
        Args:
            controlnet_name: Blur ControlNet model name
            blur_kernel_size: Size of Gaussian blur kernel
            **kwargs: Additional parameters for ControlNetTransformer
        """
        super().__init__(controlnet_name=controlnet_name, **kwargs)
        self.blur_kernel_size = blur_kernel_size
    
    def preprocess_control_image(self, image: Image.Image) -> Image.Image:
        """Generate blurred image for control."""
        return get_blur_map(image, kernel_size=self.blur_kernel_size)


class TransformerManager:
    """
    Manages multiple transformer models.
    
    This class provides a unified interface to initialize, use, and switch between
    different transformer models.
    """
    
    def __init__(self, default_model: str = "sd35", **kwargs):
        """
        Initialize the TransformerManager.
        
        Args:
            default_model: Default model to use ('sd35', 'canny', 'depth', 'blur')
            **kwargs: Parameters to pass to transformers
        """
        self.transformers = {}
        self.current_model = default_model
        self.kwargs = kwargs
        
        # Register available model types
        self.model_types = {
            'sd35': SD35Transformer,
            'canny': CannyControlTransformer,
            'depth': DepthControlTransformer,
            'blur': BlurControlTransformer
        }
        
        # Initialize default model
        self.initialize_model(default_model)
    
    def initialize_model(self, model_type: str) -> None:
        """
        Initialize a specific model.
        
        Args:
            model_type: Type of model to initialize
        """
        if model_type not in self.model_types:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(self.model_types.keys())}")
        
        if model_type not in self.transformers:
            logger.info(f"Initializing model: {model_type}")
            try:
                transformer_class = self.model_types[model_type]
                self.transformers[model_type] = transformer_class(**self.kwargs)
                logger.info(f"Model {model_type} initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing model {model_type}: {str(e)}")
                raise RuntimeError(f"Failed to initialize model {model_type}: {str(e)}")
    
    def get_transformer(self, model_type: Optional[str] = None) -> BaseTransformer:
        """
        Get a transformer by type.
        
        Args:
            model_type: Type of model to get (uses current model if None)
            
        Returns:
            Initialized transformer
        """
        if model_type is None:
            model_type = self.current_model
        
        if model_type not in self.transformers:
            self.initialize_model(model_type)
        
        return self.transformers[model_type]
    
    def set_current_model(self, model_type: str) -> None:
        """
        Set the current model.
        
        Args:
            model_type: Type of model to set as current
        """
        if model_type not in self.model_types:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(self.model_types.keys())}")
        
        self.current_model = model_type
        if model_type not in self.transformers:
            self.initialize_model(model_type)
    
    def transform(self, image: Image.Image, prompt: str, model_type: Optional[str] = None, **kwargs) -> Image.Image:
        """
        Transform image using specified or current model.
        
        Args:
            image: Input image
            prompt: Text prompt
            model_type: Type of model to use (uses current model if None)
            **kwargs: Additional parameters for transformation
            
        Returns:
            Transformed image
        """
        transformer = self.get_transformer(model_type)
        return transformer.transform(image, prompt, **kwargs)
    
    def cleanup(self) -> None:
        """Clean up all transformers to free memory."""
        for model_type, transformer in self.transformers.items():
            logger.info(f"Cleaning up model: {model_type}")
            transformer.cleanup()
        
        # Clear transformer cache
        self.transformers = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()