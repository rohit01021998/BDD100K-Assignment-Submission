"""Add bbox_area field to each detection for sorting/filtering in FiftyOne.

Usage:
    python -m yolo11s_eval.add_bbox_area
"""

import fiftyone as fo

from .config import FIFTYONE_DATASET_NAME


def main() -> None:
    """Compute and store bbox area for GT and prediction detections."""
    if not fo.dataset_exists(FIFTYONE_DATASET_NAME):
        print(f"Dataset '{FIFTYONE_DATASET_NAME}' not found.")
        return

    dataset = fo.load_dataset(FIFTYONE_DATASET_NAME)
    print(f"Adding bbox_area to {len(dataset)} samples ...")

    for sample in dataset:
        # GT detections
        if sample.detections:
            for det in sample.detections.detections:
                _, _, w, h = det.bounding_box  # relative coords
                det["bbox_area"] = round(w * h, 6)
            sample.save()

        # Prediction detections
        if sample.yolo11s:
            for det in sample.yolo11s.detections:
                _, _, w, h = det.bounding_box
                det["bbox_area"] = round(w * h, 6)
            sample.save()

    print("Done. You can now sort/filter by bbox_area in the sidebar.")
    print("Restart fo_launch to see it.")


if __name__ == "__main__":
    main()
