"""YOLO11s Inference Script.

Runs YOLO11s on the BDD100K validation set, maps the 13 training
classes to the 10 evaluation classes, and saves predictions in
COCO-format JSON for downstream evaluation.

Usage:
    # Full validation set (default)
    python -m yolo11s_eval.inference

    # Single image / video / directory
    python -m yolo11s_eval.inference --source /path/to/image.jpg

    # Custom confidence threshold
    python -m yolo11s_eval.inference --conf 0.1
"""

import argparse
import json
import os

from tqdm import tqdm
from ultralytics import YOLO

from .config import (
    EVAL_NAME_TO_ID,
    PRED_JSON,
    VAL_IMG_DIR,
    WEIGHTS_PATH,
    YOLO_TO_EVAL,
)


def run_val_inference(
    weights_path: str,
    val_img_dir: str,
    output_json: str,
    conf_thres: float = 0.25,
    max_det: int = 300,
) -> None:
    """Run inference on every image in *val_img_dir*.

    Maps YOLO 13-class predictions to the 10-class evaluation schema
    and writes a COCO-format results JSON to *output_json*.
    """
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = sorted(
        f
        for f in os.listdir(val_img_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
        and not f.startswith(".")
    )
    print(f"Found {len(img_files)} images in {val_img_dir}")

    # Build file_name → image_id mapping from COCO annotation
    gt_json_path = os.path.join(val_img_dir, "_annotations.coco.json")
    if os.path.exists(gt_json_path):
        with open(gt_json_path, "r") as fh:
            gt = json.load(fh)
        fname_to_id = {img["file_name"]: img["id"] for img in gt["images"]}
        print(f"Loaded image-ID mapping from {gt_json_path}")
    else:
        fname_to_id = {f: i for i, f in enumerate(img_files)}
        print("Warning: annotation file not found; using index as image_id")

    coco_results: list[dict] = []
    total_detections = 0

    print(f"\nRunning inference (conf={conf_thres}, max_det={max_det})...")
    for fname in tqdm(img_files, desc="Inference"):
        filepath = os.path.join(val_img_dir, fname)
        image_id = fname_to_id.get(fname)
        if image_id is None:
            continue

        results = model.predict(
            source=filepath, conf=conf_thres,
            max_det=max_det, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            w, h = x2 - x1, y2 - y1
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            eval_name = YOLO_TO_EVAL.get(cls_id)
            if eval_name is None:
                continue

            coco_results.append({
                "image_id": image_id,
                "category_id": EVAL_NAME_TO_ID[eval_name],
                "bbox": [round(x1, 2), round(y1, 2),
                         round(w, 2), round(h, 2)],
                "score": round(conf, 4),
            })
            total_detections += 1

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


def run_simple_inference(
    weights_path: str,
    source,
    save_dir: str = "inference_results",
    conf_thres: float = 0.25,
    save: bool = True,
    show: bool = False,
) -> None:
    """Run inference on a single image / video / directory / webcam."""
    print(f"Loading YOLO model from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"Error: weights not found at {weights_path}")
        return

    model = YOLO(weights_path)
    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Running inference on: {source}")
    model.predict(
        source=source, conf=conf_thres, save=save,
        project=save_dir, name="predict", exist_ok=True, show=show,
    )
    print("Inference completed.")


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="YOLO11s Inference")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--val-dir", type=str, default=VAL_IMG_DIR,
                        help="Validation image directory (COCO format).")
    parser.add_argument("--output-json", type=str, default=PRED_JSON,
                        help="Output path for COCO-format predictions.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--source", type=str, default=None,
                        help="Image/video/dir for simple inference mode.")
    parser.add_argument("--save-dir", type=str, default="inference_results")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.source:
        src = int(args.source) if args.source.isdigit() else args.source
        run_simple_inference(
            weights_path=args.weights, source=src,
            save_dir=args.save_dir, conf_thres=args.conf,
            save=not args.no_save, show=args.show,
        )
    else:
        run_val_inference(
            weights_path=args.weights,
            val_img_dir=args.val_dir,
            output_json=args.output_json,
            conf_thres=args.conf,
        )


if __name__ == "__main__":
    main()
