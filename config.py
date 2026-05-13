# config.py — smart scanner settings

# Camera
CAMERA_INDEX  = 1
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# Blueprint overlay
OVERLAY_ALPHA = 0.45

# Phase thresholds
DETECTION_THRESHOLD   = 0.40   # shape score to confirm module identity
ALIGNMENT_THRESHOLD   = 0.75   # shape score to trigger capture
ALIGNMENT_HOLD_FRAMES = 10     # frames to hold alignment before capture
FILL_THRESHOLD        = 0.60   # module must fill 60% of blueprint box
ANGLE_TOLERANCE       = 10.0   # not used for capture — kept for display only
CAMERA_TILT_TOLERANCE = 0.15   # aspect ratio tolerance for 90° camera check
                                # 0.0 = perfect, 0.15 = slight tilt allowed

# Feature detection
CIRCULARITY_THRESHOLD = 0.75
MIN_FEATURE_AREA      = 100

# Paths
BLUEPRINTS_DIR = "blueprints/"
SPECS_FILE     = "specs/specs.json"

# Display
WINDOW_NAME = "Smart Module Scanner"
