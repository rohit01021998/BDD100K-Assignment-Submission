"""Shared configuration for RF-DETR evaluation scripts.

All paths are derived from BASE_DIR (project root) and the script
location so the scripts work regardless of the current directory.
"""

import os

# Project root: parent of the rf_detr_eval/ folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# ── Dataset paths ───────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(BASE_DIR, "rf_detr_medium_thor_3x_ms_final.pth")
VAL_IMG_DIR = os.path.join(BASE_DIR, "coco_format_full", "valid")
GT_ANN_PATH = os.path.join(VAL_IMG_DIR, "_annotations.coco.json")
PRED_JSON = os.path.join(SCRIPT_DIR, "rf_detr_val_predictions.json")
BDD_LABELS = os.path.join(
    BASE_DIR,
    "assignment_data_bdd",
    "bdd100k_labels_release",
    "bdd100k",
    "labels",
    "bdd100k_labels_images_val.json",
)
OUT_DIR = os.path.join(BASE_DIR, "rf_detr_evaluation_output")

# ── RF-DETR model settings ──────────────────────────────────────────
NUM_CLASSES = 10
RESOLUTION = 640
CONFIDENCE_THRESHOLD = 0.25

# ── 10 evaluation classes (same order as COCO annotation categories) ─
EVAL_10_CLASSES = [
    "bike", "traffic sign", "train", "traffic light",
    "motor", "truck", "person", "rider", "bus", "car",
]

EVAL_NAME_TO_ID = {name: i + 1 for i, name in enumerate(EVAL_10_CLASSES)}

# ── FiftyOne settings ───────────────────────────────────────────────
FIFTYONE_DATASET_NAME = "bdd100k_rf_detr_eval"
FIFTYONE_PORT = 5152
