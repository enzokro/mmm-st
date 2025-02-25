#!/usr/bin/env python3
"""
Real-time video transformation using SD 3.5 with multiple transformer options.

This implementation uses FastHTML with MonsterUI for a modern, responsive UI.
Key features:
- Multiple transformer models (SD3.5 Turbo, ControlNets for depth/canny/blur)
- WebSockets for real-time video streaming
- Component-based architecture with clear separation of concerns
"""

import os
import gc
import time
import atexit
import base64
import logging
import asyncio
import threading
from io import BytesIO

# Import FastHTML and MonsterUI components
from fasthtml.common import *
from monsterui.all import *

# Import transformer models
from tfm import TransformerManager, SD35Transformer

# Import utilities and config
from utils import SharedResources, VideoProcessingThread, convert_to_pil_image
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastHTML app with MonsterUI theme
app, rt = fast_app(
    hdrs=Theme.blue.headers(),
    title="SD 3.5 Video Transformer",
    static_path="static",  # Directory for static files
)

# Initialize shared resources from utils.py
shared_resources = SharedResources()

# Initialize model manager from tfm.py
try:
    transformer_manager = TransformerManager(
        default_model=Config.DEFAULT_MODEL,
        use_quantization=Config.USE_QUANTIZATION,
        device=Config.DEVICE,
        dtype=Config.DTYPE,
        negative_prompt=Config.NEGATIVE_PROMPT
    )
    # Assign to shared resources for thread access
    shared_resources.transformer_manager = transformer_manager
    logger.info(f"TransformerManager initialized with default model: {Config.DEFAULT_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize transformer manager: {str(e)}")

# Start video processing thread using the updated class from utils.py
video_thread = VideoProcessingThread(shared_resources)
video_thread.start()


# UI Components
def index_page():
    """Build the main UI using MonsterUI components."""
    return Container(
        Grid(
            # Video feed card
            Card(
                Img(id="videoFeed", src="", alt="Video Feed", cls="w-full"),
                header=H3("SD 3.5 Video Transformer"),
                footer=Div(id="status", cls=TextPresets.muted_sm)("Connecting...")
            ),
            # Controls card
            Card(
                Form(
                    LabelInput("Prompt", id="promptInput", placeholder="Enter a prompt to transform the video..."),
                    H4("Model Selection", cls="mt-4"),
                    # Model selector using MonsterUI components
                    Grid(
                        *[Button(model.upper(), id=f"model-{model}", 
                               cls=(ButtonT.primary if model == Config.DEFAULT_MODEL else ButtonT.secondary),
                               hx_post=f"/api/set_model/{model}") 
                          for model in Config.MODEL_TYPES],
                        cols=2,
                        cls="mb-4"
                    ),
                    DivFullySpaced(
                        Button("Set Prompt", id="setPromptBtn", cls=ButtonT.primary, 
                               hx_post="/api/set_prompt", hx_include="#promptInput"),
                        Button("Refresh", id="refreshBtn", cls=ButtonT.secondary, 
                               hx_post="/api/refresh_generator")
                    )
                ),
                header=H3("Controls")
            ),
            # Stats card
            Card(
                Div(id="statsContent", cls="space-y-2")(
                    DivHStacked(P("Device:"), P(id="device", cls=TextPresets.bold_sm)),
                    DivHStacked(P("Model:"), P(id="model", cls=TextPresets.bold_sm)),
                    DivHStacked(P("Frame Size:"), P(id="frameSize", cls=TextPresets.bold_sm)),
                    DivHStacked(P("Steps:"), P(id="steps", cls=TextPresets.bold_sm)),
                    DivHStacked(P("Connected Clients:"), P(id="connectedClients", cls=TextPresets.bold_sm)),
                    DivHStacked(P("Memory Used:"), P(id="memoryUsed", cls=TextPresets.bold_sm)),
                ),
                header=H3("Stats"),
                hx_get="/api/stats",
                hx_trigger="load, every 5s"
            ),
            # Add model info card
            Card(
                Div(id="modelInfoContent")(
                    H4("SD3.5 Turbo", cls=TextPresets.bold_lg),
                    P("Fast model optimized for 4-step inference", cls=TextPresets.muted_sm),
                    DividerLine(),
                    Strong("Canny ControlNet"),
                    P("Uses edge detection to guide generation", cls=TextPresets.muted_sm),
                    DividerLine(),
                    Strong("Depth ControlNet"),
                    P("Uses depth maps for 3D-aware generation", cls=TextPresets.muted_sm),
                    DividerLine(),
                    Strong("Blur ControlNet"),
                    P("Uses blurred images for upscaling", cls=TextPresets.muted_sm),
                ),
                header=H3("Model Information"),
                hx_get=f"/api/model_info/{Config.DEFAULT_MODEL}",
                hx_trigger="load"
            ),
            cols=Config.UI_COLUMNS,
            cls="gap-4"
        ),
        # WebSocket client script 
        Script("""
document.addEventListener('DOMContentLoaded', function() {
    const videoFeed = document.getElementById('videoFeed');
    const status = document.getElementById('status');
    
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = function() {
            status.textContent = 'Connected';
            status.classList.add('text-success');
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'frame') {
                videoFeed.src = `data:image/jpeg;base64,${data.data}`;
            } else if (data.type === 'connected') {
                status.textContent = data.message;
                status.classList.add('text-success');
            }
        };
        
        ws.onclose = function() {
            status.textContent = 'Disconnected. Reconnecting...';
            status.classList.remove('text-success');
            status.classList.add('text-error');
            setTimeout(connectWebSocket, 1000);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
            status.textContent = 'Connection error';
            status.classList.remove('text-success');
            status.classList.add('text-error');
        };

        // Set event listeners for model buttons
        document.querySelectorAll('[id^="model-"]').forEach(button => {
            button.addEventListener('click', function() {
                // Reset all buttons to secondary
                document.querySelectorAll('[id^="model-"]').forEach(btn => {
                    btn.classList.remove('uk-btn-primary');
                    btn.classList.add('uk-btn-secondary');
                });
                // Set clicked button to primary
                this.classList.remove('uk-btn-secondary');
                this.classList.add('uk-btn-primary');
            });
        });

        // Set custom event listeners for HTMX endpoints
        document.addEventListener('htmx:afterRequest', function(evt) {
            if (evt.detail.pathInfo.requestPath === '/api/set_prompt') {
                const promptInput = document.getElementById('promptInput');
                if (evt.detail.successful) {
                    status.textContent = `Prompt set: ${promptInput.value}`;
                    status.classList.add('text-success');
                    status.classList.remove('text-error');
                } else {
                    status.textContent = 'Error setting prompt';
                    status.classList.remove('text-success');
                    status.classList.add('text-error');
                }
            }
            
            if (evt.detail.pathInfo.requestPath === '/api/refresh_generator') {
                if (evt.detail.successful) {
                    status.textContent = 'Generator refreshed';
                    status.classList.add('text-success');
                    status.classList.remove('text-error');
                } else {
                    status.textContent = 'Error refreshing generator';
                    status.classList.remove('text-success');
                    status.classList.add('text-error');
                }
            }
            
            if (evt.detail.pathInfo.requestPath.startsWith('/api/set_model/')) {
                if (evt.detail.successful) {
                    const model = evt.detail.pathInfo.requestPath.split('/').pop();
                    status.textContent = `Model set to: ${model.toUpperCase()}`;
                    status.classList.add('text-success');
                    status.classList.remove('text-error');
                    
                    // Update model info
                    htmx.ajax('GET', `/api/model_info/${model}`, '#modelInfoContent');
                } else {
                    status.textContent = 'Error setting model';
                    status.classList.remove('text-success');
                    status.classList.add('text-error');
                }
            }
        });
    }

    // Connect WebSocket when page loads
    connectWebSocket();
});
        """)
    )


# FastHTML routes
@rt("/")
def get():
    """Render the main page."""
    return Titled("SD 3.5 Video Transformer", index_page())


@rt("/api/set_prompt")
def post(prompt: str):
    """Set the current prompt for image transformation."""
    try:
        if not prompt:
            return {"error": "Prompt is required"}, 400
        
        shared_resources.update_prompt(prompt)
        return {"message": "Prompt set successfully"}
    except Exception as e:
        logger.error(f"Error setting prompt: {str(e)}")
        return {"error": f"Could not set prompt: {str(e)}"}, 500


@rt("/api/set_model/{model_type}")
def post(model_type: str):
    """Set the current model type."""
    try:
        if model_type not in Config.MODEL_TYPES:
            return {"error": f"Unknown model type: {model_type}"}, 400
        
        shared_resources.set_model(model_type)
        return {"message": f"Model set to: {model_type}"}
    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        return {"error": f"Could not set model: {str(e)}"}, 500


@rt("/api/refresh_generator")
def post():
    """Refresh the generator for the diffusion model."""
    try:
        # Reset the generator to get fresh randomness
        with shared_resources.lock:
            if shared_resources.transformer_manager:
                # Get the current model's transformer
                transformer = shared_resources.transformer_manager.get_transformer()
                # Reset its generator with a new seed based on current time
                transformer.generator = torch.Generator(device="cpu").manual_seed(int(time.time()))
        
        return {"message": "Generator refreshed"}
    except Exception as e:
        logger.error(f"Error refreshing generator: {str(e)}")
        return {"error": f"Could not refresh generator: {str(e)}"}, 500


@rt("/api/stats")
def get():
    """Get system statistics."""
    try:
        # Get model and device info from transformer manager
        current_model = shared_resources.get_model()
        device = Config.DEVICE
        
        # Get model-specific stats
        if shared_resources.transformer_manager:
            transformer = shared_resources.transformer_manager.get_transformer()
            steps = getattr(transformer, 'num_inference_steps', Config.NUM_STEPS)
        else:
            steps = Config.NUM_STEPS
        
        # Memory info for CUDA
        memory_used = "N/A"
        if torch.cuda.is_available():
            memory_used = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        
        return Div(id="statsContent", cls="space-y-2")(
            DivHStacked(P("Device:"), P(device, cls=TextPresets.bold_sm)),
            DivHStacked(P("Model:"), P(current_model.upper(), cls=TextPresets.bold_sm)),
            DivHStacked(P("Frame Size:"), P(f"{Config.WIDTH}x{Config.HEIGHT}", cls=TextPresets.bold_sm)),
            DivHStacked(P("Steps:"), P(str(steps), cls=TextPresets.bold_sm)),
            DivHStacked(P("Connected Clients:"), P(str(shared_resources.connected_clients), cls=TextPresets.bold_sm)),
            DivHStacked(P("Memory Used:"), P(memory_used, cls=TextPresets.bold_sm)),
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return Div(f"Could not get stats: {str(e)}", cls="text-error")


@rt("/api/model_info/{model_type}")
def get(model_type: str):
    """Get information about a specific model."""
    try:
        model_info = {
            "sd35": {
                "title": "SD3.5 Turbo",
                "description": "Fast model optimized for 4-step inference",
                "steps": Config.NUM_STEPS,
                "guidance": Config.CFG
            },
            "canny": {
                "title": "Canny ControlNet",
                "description": "Uses edge detection to guide generation",
                "steps": Config.CONTROLNET_STEPS,
                "guidance": Config.CONTROLNET_CFG
            },
            "depth": {
                "title": "Depth ControlNet",
                "description": "Uses depth maps for 3D-aware generation",
                "steps": Config.CONTROLNET_STEPS,
                "guidance": Config.CONTROLNET_CFG
            },
            "blur": {
                "title": "Blur ControlNet",
                "description": "Uses blurred images for upscaling",
                "steps": Config.CONTROLNET_STEPS,
                "guidance": Config.CONTROLNET_CFG
            }
        }
        
        if model_type not in model_info:
            return {"error": f"Unknown model type: {model_type}"}, 400
        
        info = model_info[model_type]
        
        return Div(id="modelInfoContent")(
            H4(info["title"], cls=TextPresets.bold_lg),
            P(info["description"], cls=TextPresets.muted_sm),
            DividerLine(),
            Grid(
                Div(Strong("Steps:"), P(str(info["steps"]), cls=TextPresets.muted_sm)),
                Div(Strong("Guidance:"), P(str(info["guidance"]), cls=TextPresets.muted_sm)),
                cols=2
            ),
            # Other model details could be added here
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return Div(f"Could not get model info: {str(e)}", cls="text-error")


# WebSocket endpoint for streaming frames
@app.ws("/ws")
async def ws(msg: str, send):
    """Stream frames to the client via WebSocket."""
    # Increment client count
    client_id = shared_resources.increment_clients()
    logger.info(f"Client {client_id} connected. Total clients: {shared_resources.connected_clients}")
    
    try:
        # Send initial connection confirmation
        await send({"type": "connected", "message": "Connection established"})
        
        frame_count = 0
        while not shared_resources.stop_event.is_set():
            frame = shared_resources.get_frame()
            if frame:
                frame_count += 1
                # Only send every nth frame to reduce bandwidth
                frame_skip = getattr(Config, 'FRAME_SKIP', 2)
                if frame_count % frame_skip == 0:
                    img_byte_arr = BytesIO()
                    frame = convert_to_pil_image(frame)
                    display_width = getattr(Config, 'DISPLAY_WIDTH', 640)
                    display_height = getattr(Config, 'DISPLAY_HEIGHT', 480)
                    frame = frame.resize((display_width, display_height))
                    frame.save(img_byte_arr, format='JPEG', quality=85)
                    img_byte_arr.seek(0)
                    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    await send({
                        "type": "frame",
                        "data": encoded_img
                    })
            
            # Control frame rate
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Decrement client count
        shared_resources.decrement_clients()
        logger.info(f"Client {client_id} disconnected. Total clients: {shared_resources.connected_clients}")


def cleanup():
    """Clean up resources on application shutdown."""
    logger.info("Application is shutting down. Cleaning up resources.")
    shared_resources.stop_event.set()
    video_thread.join(timeout=5.0)
    
    # Clean up transformer resources
    if shared_resources.transformer_manager:
        shared_resources.transformer_manager.cleanup()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Register cleanup handler
atexit.register(cleanup)

# Start the application
serve(port=Config.PORT, host=Config.HOST, reload=False)
