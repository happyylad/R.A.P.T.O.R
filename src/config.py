# File: src/config.py

"""
R.A.P.T.O.R. System Configuration
This file holds key system-wide variables and settings.
"""

# --- Model Configuration ---

# Path to the custom-trained YOLOv8 model weights, relative to the project root.
MODEL_PATH = 'models/raptor_custom_v1.pt'

# Class names that the model was trained on.
# The order of this list MUST EXACTLY MATCH the 'names' list in the training YAML.
RAPTOR_CLASSES = [
    'person',
    'plane',
    'helicopter',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'structure'
]