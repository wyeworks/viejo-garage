import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "output"

YOLO_MODEL = "yolov8l.pt"  # Using larger model for better detection
YOLO_CONFIDENCE = 0.1  # Low threshold for enhanced detection
YOLO_IOU_THRESHOLD = 0.45

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

PARKING_SPACE_IOU_THRESHOLD = 0.3

OVERLAY_CONFIG = {
    "empty_color": (0, 255, 0),      # Green for empty spaces
    "occupied_color": (0, 0, 255),   # Red for occupied spaces  
    "pending_color": (0, 255, 255),  # Yellow for pending/processing
    "box_thickness": 2,
    "font_scale": 0.6,
    "font_thickness": 2
}

DEBUG = os.getenv("DEBUG", "False").lower() == "true"