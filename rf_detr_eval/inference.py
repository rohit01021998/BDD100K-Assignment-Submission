"""RF-DETR Inference Script.

Runs RF-DETR Medium on the BDD100K validation set and saves
predictions in COCO-format JSON for downstream evaluation.

Usage:
    # Full validation set (default)
    python -m rf_detr_eval.inference

    # Custom confidence threshold
    python -m rf_detr_eval.inference --conf 0.001
"""

import argparse
import json
import os

import torch
from PIL import Image
from rfdetr import RFDETRMedium
from tqdm import tqdm

from .config import (
    CONFIDENCE_THRESHOLD,
    GT_ANN_PATH,
    NUM_CLASSES,
    PRED_JSON,
    RESOLUTION,
    VAL_IMG_DIR,
    WEIGHTS_PATH,
)


def load_model(weights_path: str) -> RFDETRMedium:
    """Load RF-DETR Medium model with trained weights."""
    print(f"Loading RF-DETR model (num_classes={NUM_CLASSES}, res={RESOLUTION}) ...")
    model = RFDETRMedium(num_classes=NUM_CLASSES, resolution=RESOLUTION)

    # Reinitialize detection head to match checkpoint (num_classes + 1 background)
    model.model.reinitialize_detection_head(NUM_CLASSES + 1)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model_state = torch.load(weights_path, map_location=device, weights_only=False)
    model.model.model.load_state_dict(model_state["model"])
    print(f"✓ Model weights loaded from {weights_path}")
    print(f"✓ Using device: {device}")
    return model


def run_val_inference(
    model,
    val_img_dir: str,
    output_json: str,
    conf_thres: float = 0.001,
) -> None:
    """Run inference on every image in *val_img_dir*.

    RF-DETR directly outputs 10-class detections (1-indexed categories).
    Writes a COCO-format results JSON to *output_json*.
    """
    print(f"\nLoading COCO annotations from: {GT_ANN_PATH}")
    with open(GT_ANN_PATH, "r") as fh:
        gt = json.load(fh)
    fname_to_id = {img["file_name"]: img["id"] for img in gt["images"]}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = sorted(
        f
        for f in os.listdir(val_img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
        and not f.startswith(".")
    )
    print(f"Found {len(img_files)} images in {val_img_dir}")

    coco_results: list[dict] = []
    total_detections = 0

    print(f"\nRunning inference (conf={conf_thres}) ...")
    for fname in tqdm(img_files, desc="Inference"):
        filepath = os.path.join(val_img_dir, fname)
        image_id = fname_to_id.get(fname)
        if image_id is None:
            continue

        try:
            img = Image.open(filepath).convert("RGB")
            dets = model.predict(img, threshold=conf_thres)

            for i in range(len(dets.xyxy)):
                x1, y1, x2, y2 = dets.xyxy[i]
                w, h = float(x2 - x1), float(y2 - y1)
                # RF-DETR class_id is 0-indexed; COCO uses 1-indexed
                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(dets.class_id[i]) + 1,
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(w, 2), round(h, 2)],
                    "score": round(float(dets.confidence[i]), 4),
                })
                total_detections += 1
        except Exception as e:
            print(f"⚠ Error processing image {image_id} ({fname}): {e}")
            continue

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(coco_results, fh, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved {total_detections} detections to {output_json}")
    print(f"  Images processed : {len(img_files)}")
    print(f"  Avg dets / image : {total_detections / max(len(img_files), 1):.1f}")
    print(f"{'=' * 60}")


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="RF-DETR Inference")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--val-dir", type=str, default=VAL_IMG_DIR,
                        help="Validation image directory (COCO format).")
    parser.add_argument("--output-json", type=str, default=PRED_JSON,
                        help="Output path for COCO-format predictions.")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    model = load_model(args.weights)
    run_val_inference(
        model,
        val_img_dir=args.val_dir,
        output_json=args.output_json,
        conf_thres=args.conf,
    )


if __name__ == "__main__":
    main()
