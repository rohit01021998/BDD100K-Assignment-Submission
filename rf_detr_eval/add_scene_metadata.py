"""Add BDD100K scene attributes (timeofday, weather, scene) to FiftyOne dataset.

Run once after voxel51_eval.py to enrich the dataset with metadata.

Usage:
    python -m rf_detr_eval.add_scene_metadata
"""

import json
import os

import fiftyone as fo

from .config import BDD_LABELS, FIFTYONE_DATASET_NAME


def main() -> None:
    """Add scene attributes to each sample in the dataset."""
    if not fo.dataset_exists(FIFTYONE_DATASET_NAME):
        print(f"Dataset '{FIFTYONE_DATASET_NAME}' not found.")
        print("Run  python -m rf_detr_eval.voxel51_eval  first.")
        return

    if not os.path.exists(BDD_LABELS):
        print(f"BDD labels not found at {BDD_LABELS}")
        return

    # Load BDD metadata
    print("Loading BDD100K scene attributes ...")
    with open(BDD_LABELS) as fh:
        bdd_data = json.load(fh)

    fname_to_attrs = {}
    for item in bdd_data:
        attrs = item.get("attributes", {})
        fname_to_attrs[item["name"]] = {
            "timeofday": attrs.get("timeofday", "unknown"),
            "weather": attrs.get("weather", "unknown"),
            "scene": attrs.get("scene", "unknown"),
        }
    print(f"  Loaded attributes for {len(fname_to_attrs)} images.")

    # Add to dataset
    dataset = fo.load_dataset(FIFTYONE_DATASET_NAME)
    updated = 0
    for sample in dataset:
        fname = os.path.basename(sample.filepath)
        attrs = fname_to_attrs.get(fname)
        if attrs:
            sample["timeofday"] = attrs["timeofday"]
            sample["weather"] = attrs["weather"]
            sample["scene"] = attrs["scene"]
            sample.save()
            updated += 1

    print(f"  Updated {updated} / {len(dataset)} samples.")
    print("  Restart fo_launch.py to see the new fields in the sidebar.")


if __name__ == "__main__":
    main()
