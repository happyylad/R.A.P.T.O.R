# File: src/config.py
# Central configuration for the R.A.P.T.O.R. project.

# This list defines the object classes for our custom-trained model.
# The order is critical and must match the training configuration.
RAPTOR_MODEL_CLASSES = [
    "person",
    "plane",
    "helicopter",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "structure",
]

# Color palette for these classes here.
RAPTOR_CLASS_COLORS = {
    "person": (0, 255, 0),  # Green
    "plane": (255, 0, 0),  # Blue
    "helicopter": (0, 0, 255),  # Red
    "small-vehicle": (255, 255, 0),  # Cyan
    "large-vehicle": (255, 0, 255),  # Magenta
    "ship": (0, 255, 255),  # Yellow
    "structure": (128, 128, 128),  # Grey
}

MODELS = {
    "Aerial Mode (R.A.P.T.O.R. v1)": "models/raptor_v1.pt",
    "Ground Mode (Default YOLOv8n)": "models/raptor_v0.pt",
}
