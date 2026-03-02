"""FiftyOne (Voxel51) evaluation — load data & evaluate.

Loads COCO ground-truth and RF-DETR predictions into FiftyOne,
runs mAP evaluation, and prints the report.  Does **not** launch
the app; use ``fo_launch.py`` for that.

Usage:
    python -m rf_detr_eval.voxel51_eval
"""

import json
import os

import fiftyone as fo

from .config import (
    FIFTYONE_DATASET_NAME,
    GT_ANN_PATH,
    OUT_DIR,
    PRED_JSON,
    VAL_IMG_DIR,
)


def main(
    val_img_dir: str = VAL_IMG_DIR,
    val_ann_path: str = GT_ANN_PATH,
    predictions_json: str = PRED_JSON,
    dataset_name: str = FIFTYONE_DATASET_NAME,
) -> None:
    """Import data, attach predictions, evaluate, then exit."""
    # 1. Delete existing dataset if present
    if fo.dataset_exists(dataset_name):
        print(f"Deleting existing dataset '{dataset_name}' ...")
        fo.delete_dataset(dataset_name)

    # 2. Load ground truth
    print(f"Loading ground truth from {val_ann_path} ...")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=val_img_dir,
        labels_path=val_ann_path,
        name=dataset_name,
    )
    print(f"  Loaded {len(dataset)} samples with ground truth.")
    dataset.persistent = True

    # 3. Load predictions
    print(f"\nLoading predictions from {predictions_json} ...")
    with open(predictions_json, "r") as fh:
        preds = json.load(fh)
    with open(val_ann_path, "r") as fh:
        gt_data = json.load(fh)

    id_to_fname = {i["id"]: i["file_name"] for i in gt_data["images"]}
    id_to_size = {i["id"]: (i["width"], i["height"]) for i in gt_data["images"]}
    cid_to_name = {c["id"]: c["name"] for c in gt_data["categories"]}

    preds_by_img: dict[int, list] = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    fp_to_sample = {
        os.path.basename(s.filepath): s for s in dataset
    }

    added = 0
    for img_id, img_preds in preds_by_img.items():
        fname = id_to_fname.get(img_id)
        if not fname or fname not in fp_to_sample:
            continue
        sample = fp_to_sample[fname]
        w, h = id_to_size[img_id]
        detections = [
            fo.Detection(
                label=cid_to_name.get(p["category_id"], "unknown"),
                bounding_box=[p["bbox"][0] / w, p["bbox"][1] / h,
                              p["bbox"][2] / w, p["bbox"][3] / h],
                confidence=p["score"],
            )
            for p in img_preds
        ]
        sample["rf_detr"] = fo.Detections(detections=detections)
        sample.save()
        added += 1
    print(f"  Added predictions to {added} samples.")

    # 4. Evaluate (class-wise, standard COCO)
    print("\nEvaluating predictions vs ground truth (COCO mAP) ...")
    results = dataset.evaluate_detections(
        "rf_detr", gt_field="detections",
        eval_key="eval", compute_mAP=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (RF-DETR on BDD100K val)")
    print("=" * 60)
    results.print_report()
    print(f"\nmAP = {results.mAP():.4f}")

    # 5. Confusion matrix
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\nGenerating confusion matrix ...")

    # Interactive HTML (plotly) — open in browser to hover/zoom
    try:
        cm_plotly = results.plot_confusion_matrix(backend="plotly")
        html_path = os.path.join(OUT_DIR, "confusion_matrix_interactive.html")
        cm_plotly.write_html(html_path)
        print(f"  Interactive (HTML): {html_path}")
    except Exception as exc:
        print(f"  Plotly interactive skipped: {exc}")

    # Static PNG (matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm_fig = results.plot_confusion_matrix(backend="matplotlib")
        png_path = os.path.join(OUT_DIR, "confusion_matrix_fo.png")
        cm_fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(cm_fig)
        print(f"  Static (PNG):       {png_path}")
    except Exception as exc:
        print(f"  Matplotlib static skipped: {exc}")

    # 6. Class-agnostic evaluation (matches boxes by IoU only)
    print("\nRunning class-agnostic evaluation (classwise=False) ...")
    agnostic = dataset.evaluate_detections(
        "rf_detr", gt_field="detections",
        eval_key="eval_agnostic",
        classwise=False, compute_mAP=True,
    )
    print("  Class-agnostic report:")
    agnostic.print_report()
    print(f"  Class-agnostic mAP = {agnostic.mAP():.4f}")

    # 7. Done
    print("\n" + "=" * 60)
    print("Dataset loaded and evaluated in FiftyOne.")
    print("  To browse interactively, run:")
    print("    python -m rf_detr_eval.fo_launch")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
