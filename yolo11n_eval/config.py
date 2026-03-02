"""Shared configuration for YOLO11s evaluation scripts.

All paths are derived from BASE_DIR (project root) and the script
location so the scripts work regardless of the current directory.
"""

import os

# Project root: parent of the yolo11s_eval/ folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# ── Dataset paths ───────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "yolo11s-weights", "best.pt")
VAL_IMG_DIR = os.path.join(BASE_DIR, "coco_format_full", "valid")
GT_ANN_PATH = os.path.join(VAL_IMG_DIR, "_annotations.coco.json")
PRED_JSON = os.path.join(SCRIPT_DIR, "yolo11n_val_predictions.json")
BDD_LABELS = os.path.join(
    BASE_DIR,
    "assignment_data_bdd",
    "bdd100k_labels_release",
    "bdd100k",
    "labels",
    "bdd100k_labels_images_val.json",
)
OUT_DIR = os.path.join(BASE_DIR, "yolo11s_evaluation_output")

# ── YOLO 13-class → BDD100K 10-class mapping ───────────────────────
YOLO_13_CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "tl_green", "tl_red",
    "tl_yellow", "tl_none", "t_sign", "train",
]

YOLO_TO_EVAL = {
    0: "person",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bike",
    6: "motor",
    7: "traffic light",   # tl_green
    8: "traffic light",   # tl_red
    9: "traffic light",   # tl_yellow
    10: "traffic light",  # tl_none
    11: "traffic sign",   # t_sign
    12: "train",
}

EVAL_10_CLASSES = [
    "bike", "traffic sign", "train", "traffic light",
    "motor", "truck", "person", "rider", "bus", "car",
]

EVAL_NAME_TO_ID = {name: i + 1 for i, name in enumerate(EVAL_10_CLASSES)}

# ── FiftyOne settings ───────────────────────────────────────────────
FIFTYONE_DATASET_NAME = "bdd100k_yolo11s_eval"
FIFTYONE_PORT = 5151
