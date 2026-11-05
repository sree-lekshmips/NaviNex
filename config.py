"""
VisionAssist 2.0 - Configuration

Central location for project paths, model paths, and webcam settings.
This module contains only constants — no executable logic.
"""

from pathlib import Path

# --- Project directories ---------------------------------------------------

# Root directory of the project (resolved to an absolute path)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Directory containing trained model weights
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Directory containing source code modules
SRC_DIR: Path = PROJECT_ROOT / "src"

# --- Model paths -------------------------------------------------------------

# Custom-trained YOLO model (fine-tuned for VisionAssist-specific classes)
CUSTOM_YOLO: Path = MODELS_DIR / "best_v2.pt"

# Pretrained COCO YOLO model (general-purpose object detection baseline)
COCO_YOLO: Path = MODELS_DIR / "yolov8n.pt"

# MiDaS depth estimation model
MIDAS_MODEL: Path = MODELS_DIR / "dpt_swin2_tiny_256.pt"

# SegFormer walkable path segmentation model
SEGFORMER_MODEL_DIR: Path = (
    MODELS_DIR / "segformer_best" / "segformer_best"
    if (MODELS_DIR / "segformer_best" / "segformer_best" / "config.json").exists()
    else MODELS_DIR / "segformer_best"
)

# --- Webcam settings ---------------------------------------------------------

# Index of the webcam device to use (0 = default system camera)
CAMERA_INDEX: int = 0

# Captured frame width, in pixels
FRAME_WIDTH: int = 1280

# Captured frame height, in pixels
FRAME_HEIGHT: int = 720