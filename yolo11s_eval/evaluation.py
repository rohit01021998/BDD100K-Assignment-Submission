"""Comprehensive Evaluation & Visualization for YOLO11s on BDD100K.

Generates all quantitative metrics, plots, and a failure-analysis
summary.  Prerequisites: run ``inference.py`` first.

Usage:
    python -m yolo11s_eval.evaluation

Output:
    All plots and metrics are saved to ``<BASE_DIR>/evaluation_output/``.
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from .config import BDD_LABELS, GT_ANN_PATH, OUT_DIR, PRED_JSON, VAL_IMG_DIR  # noqa: E402

# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
COLORS = sns.color_palette("husl", 10)


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _compute_iou(box1: list, box2: list) -> float:
    """IoU between two COCO-format boxes ``[x, y, w, h]``."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def load_data():
    """Load COCO ground-truth and predictions."""
    print("Loading ground-truth annotations ...")
    coco_gt = COCO(GT_ANN_PATH)

    print("Loading predictions ...")
    coco_dt = coco_gt.loadRes(PRED_JSON)

    with open(GT_ANN_PATH, "r") as fh:
        gt_data = json.load(fh)
    with open(PRED_JSON, "r") as fh:
        pred_data = json.load(fh)

    cat_id_to_name = {c["id"]: c["name"] for c in gt_data["categories"]}
    class_names = [c["name"] for c in gt_data["categories"]]
    return coco_gt, coco_dt, gt_data, pred_data, cat_id_to_name, class_names


def _add_bar_labels(ax, bars, fmt=".2f", fontsize=9):
    """Add value labels on top of bar-chart bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, h + 0.01,
                f"{h:{fmt}}", ha="center", va="bottom", fontsize=fontsize,
            )


def _plot_attribute_bar(data: dict, title_suffix: str, filename: str):
    """Bar chart of mAP for a scene attribute."""
    items = sorted(data.items(), key=lambda x: x[1]["mAP@0.5"], reverse=True)
    names = [f"{k}\n(n={v['count']})" for k, v in items]
    ap50 = [v["mAP@0.5"] for _, v in items]
    ap5095 = [v["mAP@0.5:0.95"] for _, v in items]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w / 2, ap50, w, label="mAP@0.5", color="#4c72b0")
    bars2 = ax.bar(x + w / 2, ap5095, w, label="mAP@0.5:0.95", color="#dd8452")
    _add_bar_labels(ax, bars1, ".3f")
    _add_bar_labels(ax, bars2, ".3f")
    ax.set_ylabel("mAP")
    ax.set_title(f"Performance by {title_suffix} — YOLO11s")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, max(ap50) + 0.1 if ap50 else 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# 1. Official COCO Evaluation
# ════════════════════════════════════════════════════════════════════

def run_coco_eval(coco_gt, coco_dt, cat_id_to_name, class_names):
    """Run official COCO evaluation; return overall + per-class results."""
    print("\n" + "=" * 60)
    print("1. OFFICIAL COCO EVALUATION")
    print("=" * 60)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "mAP@0.5:0.95": stats[0], "mAP@0.5": stats[1],
        "mAP@0.75": stats[2], "mAP_small": stats[3],
        "mAP_medium": stats[4], "mAP_large": stats[5],
        "AR@1": stats[6], "AR@10": stats[7], "AR@100": stats[8],
        "AR_small": stats[9], "AR_medium": stats[10], "AR_large": stats[11],
    }

    per_class_ap = {}
    for cat_id in sorted(cat_id_to_name):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [cat_id]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        per_class_ap[cat_id_to_name[cat_id]] = {
            "AP@0.5:0.95": ev.stats[0], "AP@0.5": ev.stats[1],
            "AP@0.75": ev.stats[2], "AP_small": ev.stats[3],
            "AP_medium": ev.stats[4], "AP_large": ev.stats[5],
        }

    path = os.path.join(OUT_DIR, "coco_metrics.json")
    with open(path, "w") as fh:
        json.dump({"overall": metrics, "per_class": per_class_ap}, fh, indent=2)
    print(f"Saved metrics → {path}")
    return metrics, per_class_ap


# ════════════════════════════════════════════════════════════════════
# 2. Per-Class AP Bar Chart
# ════════════════════════════════════════════════════════════════════

def plot_per_class_ap(per_class_ap: dict):
    """Bar chart: AP@0.5 and AP@0.5:0.95 per class."""
    print("\n2. Per-class AP bar chart ...")
    classes = list(per_class_ap)
    ap50 = [per_class_ap[c]["AP@0.5"] for c in classes]
    ap5095 = [per_class_ap[c]["AP@0.5:0.95"] for c in classes]

    idx = np.argsort(ap50)[::-1]
    classes = [classes[i] for i in idx]
    ap50 = [ap50[i] for i in idx]
    ap5095 = [ap5095[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    w = 0.35
    b1 = ax.bar(x - w / 2, ap50, w, label="AP@0.5", color="#4c72b0", alpha=0.85)
    b2 = ax.bar(x + w / 2, ap5095, w, label="AP@0.5:0.95", color="#dd8452", alpha=0.85)
    _add_bar_labels(ax, b1)
    _add_bar_labels(ax, b2)
    mean_ap = np.mean(ap50)
    ax.axhline(mean_ap, color="red", ls="--", alpha=0.7,
               label=f"Mean AP@0.5 = {mean_ap:.3f}")
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-Class Average Precision — YOLO11s on BDD100K Val")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_class_ap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved per_class_ap.png")


# ════════════════════════════════════════════════════════════════════
# 3. Confusion Matrix
# ════════════════════════════════════════════════════════════════════

def compute_confusion_matrix(coco_gt, pred_data, cat_id_to_name,
                             class_names, iou_threshold=0.5):
    """Confusion matrix by matching detections to GT via IoU."""
    print("\n3. Confusion matrix ...")
    n = len(class_names)
    cat_ids = sorted(cat_id_to_name)
    cid_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    cm = np.zeros((n + 1, n + 1), dtype=int)

    preds_by_img = defaultdict(list)
    for p in pred_data:
        preds_by_img[p["image_id"]].append(p)

    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = preds_by_img.get(img_id, [])
        gt_matched = [False] * len(gt_anns)

        for det in dt_anns:
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gt_anns):
                if gt_matched[gi]:
                    continue
                iou = _compute_iou(det["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            dt_cls = cid_to_idx[det["category_id"]]
            if best_iou >= iou_threshold and best_gi >= 0:
                gt_matched[best_gi] = True
                gt_cls = cid_to_idx[gt_anns[best_gi]["category_id"]]
                cm[dt_cls, gt_cls] += 1
            else:
                cm[dt_cls, n] += 1

        for gi, matched in enumerate(gt_matched):
            if not matched:
                gt_cls = cid_to_idx[gt_anns[gi]["category_id"]]
                cm[n, gt_cls] += 1

    labels = class_names + ["background"]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_title("Confusion Matrix — YOLO11s on BDD100K Val (IoU>=0.5)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_10class.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrix_10class.png")
    return cm


# ════════════════════════════════════════════════════════════════════
# 4. Performance by Object Size
# ════════════════════════════════════════════════════════════════════

def plot_performance_by_size(metrics: dict):
    """mAP for small / medium / large objects."""
    print("\n4. Performance by object size ...")
    sizes = ["Small\n(<32\u00b2)", "Medium\n(32\u00b2-96\u00b2)", "Large\n(>96\u00b2)"]
    ap = [metrics["mAP_small"], metrics["mAP_medium"], metrics["mAP_large"]]
    ar = [metrics["AR_small"], metrics["AR_medium"], metrics["AR_large"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sizes))
    w = 0.35
    b1 = ax.bar(x - w / 2, ap, w, label="mAP@0.5:0.95", color="#4c72b0")
    b2 = ax.bar(x + w / 2, ar, w, label="AR@100", color="#55a868")
    _add_bar_labels(ax, b1, ".3f", 10)
    _add_bar_labels(ax, b2, ".3f", 10)
    ax.set_ylabel("Score")
    ax.set_title("Performance by Object Size — YOLO11s")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.set_ylim(0, max(max(ap), max(ar)) + 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "performance_by_size.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved performance_by_size.png")


# ════════════════════════════════════════════════════════════════════
# 5. GT vs Prediction Count
# ════════════════════════════════════════════════════════════════════

def plot_gt_vs_pred_counts(gt_data, pred_data, cat_id_to_name):
    """Compare GT and prediction counts per class."""
    print("\n5. GT vs Prediction counts ...")
    gt_cnt = Counter(cat_id_to_name[a["category_id"]]
                     for a in gt_data["annotations"])
    pr_cnt = Counter(cat_id_to_name[p["category_id"]] for p in pred_data)
    classes = sorted(gt_cnt, key=lambda c: gt_cnt[c], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    w = 0.35
    ax.bar(x - w / 2, [gt_cnt[c] for c in classes], w,
           label="Ground Truth", color="#4c72b0", alpha=0.85)
    ax.bar(x + w / 2, [pr_cnt.get(c, 0) for c in classes], w,
           label="Predictions", color="#dd8452", alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title("Ground Truth vs Prediction Counts per Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gt_vs_pred_counts.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved gt_vs_pred_counts.png")


# ════════════════════════════════════════════════════════════════════
# 6. Performance by Scene Attributes
# ════════════════════════════════════════════════════════════════════

def plot_performance_by_scene(coco_gt, coco_dt, gt_data):
    """mAP breakdown by time-of-day, weather, and scene type."""
    print("\n6. Performance by scene attributes ...")
    if not os.path.exists(BDD_LABELS):
        print("  Warning: BDD labels not found — skipping.")
        return

    with open(BDD_LABELS) as fh:
        bdd_data = json.load(fh)
    fname_attrs = {it["name"]: it.get("attributes", {}) for it in bdd_data}

    groups = {k: defaultdict(list) for k in ("timeofday", "weather", "scene")}
    for img in gt_data["images"]:
        attrs = fname_attrs.get(img["file_name"], {})
        for key in groups:
            groups[key][attrs.get(key, "unknown")].append(img["id"])

    results = {}
    for key, gdict in groups.items():
        results[key] = {}
        for val, ids in gdict.items():
            if len(ids) < 10:
                continue
            try:
                ev = COCOeval(coco_gt, coco_dt, "bbox")
                ev.params.imgIds = ids
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
                results[key][val] = {
                    "mAP@0.5:0.95": ev.stats[0],
                    "mAP@0.5": ev.stats[1],
                    "count": len(ids),
                }
            except Exception as exc:
                print(f"  Warning: {key}={val}: {exc}")

    with open(os.path.join(OUT_DIR, "scene_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    for key, title, fname in [
        ("timeofday", "Time of Day", "performance_by_timeofday.png"),
        ("weather", "Weather", "performance_by_weather.png"),
        ("scene", "Scene Type", "performance_by_scene.png"),
    ]:
        if key in results and results[key]:
            _plot_attribute_bar(results[key], title, fname)
    print("  Saved scene performance plots")


# ════════════════════════════════════════════════════════════════════
# 7. Precision-Recall Curves
# ════════════════════════════════════════════════════════════════════

def plot_precision_recall_curves(coco_gt, coco_dt, cat_id_to_name):
    """PR curves for each class at IoU = 0.5."""
    print("\n7. Precision-Recall curves ...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, cat_id in enumerate(sorted(cat_id_to_name)):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [cat_id]
        ev.evaluate()
        ev.accumulate()
        prec = ev.eval["precision"][0, :, 0, 0, 2]
        rec = np.linspace(0, 1, len(prec))
        ap = np.mean(prec[prec > -1]) if np.any(prec > -1) else 0.0

        ax = axes[idx]
        ax.plot(rec, prec, color=COLORS[idx], linewidth=2)
        ax.fill_between(rec, prec, alpha=0.2, color=COLORS[idx])
        ax.set_title(f"{cat_id_to_name[cat_id]}\nAP@0.5={ap:.3f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)

    plt.suptitle("Precision-Recall Curves (IoU=0.5) — YOLO11s",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "precision_recall_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved precision_recall_curves.png")


# ════════════════════════════════════════════════════════════════════
# 8. F1 vs Confidence Threshold
# ════════════════════════════════════════════════════════════════════

def plot_f1_vs_confidence(gt_data, pred_data):
    """Plot F1 vs confidence to find the optimal operating point."""
    print("\n8. F1 vs confidence threshold ...")
    gt_by_img = defaultdict(list)
    for ann in gt_data["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    thresholds = np.arange(0.05, 0.95, 0.05)
    f1_scores = []

    for thresh in thresholds:
        tp = fp = fn = 0
        filtered = [p for p in pred_data if p["score"] >= thresh]
        pr_by_img = defaultdict(list)
        for p in filtered:
            pr_by_img[p["image_id"]].append(p)

        for img_id in set(gt_by_img) | set(pr_by_img):
            gts = gt_by_img.get(img_id, [])
            dts = pr_by_img.get(img_id, [])
            matched = [False] * len(gts)
            for det in sorted(dts, key=lambda d: -d["score"]):
                best_iou, best_gi = 0, -1
                for gi, gt in enumerate(gts):
                    if matched[gi]:
                        continue
                    iou = _compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
                if best_iou >= 0.5 and best_gi >= 0:
                    tp += 1
                    matched[best_gi] = True
                else:
                    fp += 1
            fn += sum(1 for m in matched if not m)

        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1_scores.append(f1)

    best_i = int(np.argmax(f1_scores))
    best_t, best_f1 = thresholds[best_i], f1_scores[best_i]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1_scores, "b-o", linewidth=2, markersize=5)
    ax.axvline(best_t, color="red", ls="--", alpha=0.7,
               label=f"Best threshold = {best_t:.2f} (F1 = {best_f1:.3f})")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Confidence Threshold — YOLO11s")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "f1_vs_confidence.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Best threshold: {best_t:.2f} (F1 = {best_f1:.3f})")
    print("  Saved f1_vs_confidence.png")
    return best_t, best_f1


# ════════════════════════════════════════════════════════════════════
# 9. Qualitative Samples
# ════════════════════════════════════════════════════════════════════

def save_qualitative_samples(coco_gt, pred_data, gt_data,
                             cat_id_to_name, n_samples=10):
    """Save images with GT (green) and predictions (red) overlaid."""
    print(f"\n9. Saving {n_samples} qualitative samples ...")
    sample_dir = os.path.join(OUT_DIR, "qualitative_samples")
    os.makedirs(sample_dir, exist_ok=True)

    pr_by_img = defaultdict(list)
    for p in pred_data:
        pr_by_img[p["image_id"]].append(p)
    img_ids = sorted(pr_by_img, key=lambda x: len(pr_by_img[x]), reverse=True)
    id_to_fname = {img["id"]: img["file_name"] for img in gt_data["images"]}

    saved = 0
    for img_id in img_ids:
        if saved >= n_samples:
            break
        fname = id_to_fname.get(img_id)
        if not fname:
            continue
        path = os.path.join(VAL_IMG_DIR, fname)
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id)):
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
            draw.text((x, max(0, y - 12)),
                      f"GT:{cat_id_to_name.get(ann['category_id'], '?')}",
                      fill="lime")
        for p in pr_by_img[img_id]:
            if p["score"] < 0.3:
                continue
            x, y, w, h = p["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            draw.text((x, y + h + 2),
                      f"P:{cat_id_to_name.get(p['category_id'], '?')} "
                      f"{p['score']:.2f}",
                      fill="red")
        img.save(os.path.join(sample_dir, f"sample_{saved:02d}_{fname}"))
        saved += 1
    print(f"  Saved {saved} samples to {sample_dir}/")


# ════════════════════════════════════════════════════════════════════
# 10. Failure Analysis
# ════════════════════════════════════════════════════════════════════

def failure_analysis(per_class_ap, metrics, gt_data, pred_data, cat_id_to_name):
    """Generate a written failure-analysis summary."""
    print("\n10. Failure analysis ...")
    gt_cnt = Counter(cat_id_to_name[a["category_id"]]
                     for a in gt_data["annotations"])
    size_dist = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    for ann in gt_data["annotations"]:
        area = ann["area"]
        cn = cat_id_to_name[ann["category_id"]]
        if area < 32 ** 2:
            size_dist[cn]["small"] += 1
        elif area < 96 ** 2:
            size_dist[cn]["medium"] += 1
        else:
            size_dist[cn]["large"] += 1

    lines = [
        "=" * 60, "FAILURE ANALYSIS SUMMARY", "=" * 60, "",
        "## Overall Performance",
        f"  mAP@0.5:0.95 = {metrics['mAP@0.5:0.95']:.4f}",
        f"  mAP@0.5      = {metrics['mAP@0.5']:.4f}",
        f"  mAP@0.75     = {metrics['mAP@0.75']:.4f}", "",
        "## Performance by Object Size",
        f"  Small  (<32\u00b2px):  mAP = {metrics['mAP_small']:.4f}",
        f"  Medium (32-96\u00b2):  mAP = {metrics['mAP_medium']:.4f}",
        f"  Large  (>96\u00b2px):  mAP = {metrics['mAP_large']:.4f}", "",
        "## Per-Class Analysis (sorted by AP@0.5)",
        f"  {'Class':<16} {'AP@0.5':>8} {'AP@0.5:0.95':>12} "
        f"{'GT Count':>10} {'Small%':>8}",
        "  " + "-" * 56,
    ]
    for cn in sorted(per_class_ap, key=lambda c: per_class_ap[c]["AP@0.5"],
                     reverse=True):
        ap50 = per_class_ap[cn]["AP@0.5"]
        ap5095 = per_class_ap[cn]["AP@0.5:0.95"]
        total = sum(size_dist[cn].values())
        spct = size_dist[cn]["small"] / total * 100 if total else 0
        lines.append(f"  {cn:<16} {ap50:>8.3f} {ap5095:>12.3f} "
                     f"{gt_cnt.get(cn, 0):>10} {spct:>7.1f}%")

    lines += ["", "## Key Failure Patterns", ""]
    for cn, vals in sorted(per_class_ap.items(),
                           key=lambda x: x[1]["AP@0.5"])[:3]:
        total = sum(size_dist[cn].values())
        spct = size_dist[cn]["small"] / total * 100 if total else 0
        lines.append(f"  * {cn}: AP@0.5 = {vals['AP@0.5']:.3f}")
        lines.append(f"    GT count = {gt_cnt.get(cn, 0)}, "
                     f"Small objects = {spct:.0f}%")
        if gt_cnt.get(cn, 0) < 100:
            lines.append("    -> Very few samples. "
                         "Needs augmentation / oversampling.")
        if spct > 50:
            lines.append("    -> Majority small. "
                         "Higher resolution or FPN improvements needed.")
        lines.append("")

    lines += [
        "## Suggested Improvements",
        "  1. Data: augment underrepresented classes (train, motor, rider)",
        "  2. Data: add more night / adverse-weather samples",
        "  3. Model: use a larger backbone (YOLO11m) for small-object AP",
        "  4. Model: increase input resolution 640 -> 1280",
        "  5. Post-processing: class-specific confidence thresholds",
        "  6. Training: merge traffic-light sub-classes during training",
    ]

    summary = "\n".join(lines)
    path = os.path.join(OUT_DIR, "failure_analysis.txt")
    with open(path, "w") as fh:
        fh.write(summary)
    print(summary)
    print(f"\n  Saved {path}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    """Run the full evaluation pipeline."""
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("YOLO11s COMPREHENSIVE EVALUATION ON BDD100K")
    print("=" * 60)

    coco_gt, coco_dt, gt_data, pred_data, cid2name, cls_names = load_data()

    metrics, per_class_ap = run_coco_eval(coco_gt, coco_dt, cid2name, cls_names)
    plot_per_class_ap(per_class_ap)
    compute_confusion_matrix(coco_gt, pred_data, cid2name, cls_names)
    plot_performance_by_size(metrics)
    plot_gt_vs_pred_counts(gt_data, pred_data, cid2name)
    plot_performance_by_scene(coco_gt, coco_dt, gt_data)
    plot_precision_recall_curves(coco_gt, coco_dt, cid2name)
    plot_f1_vs_confidence(gt_data, pred_data)
    save_qualitative_samples(coco_gt, pred_data, gt_data, cid2name)
    failure_analysis(per_class_ap, metrics, gt_data, pred_data, cid2name)

    print(f"\n{'=' * 60}")
    print(f"ALL PLOTS AND METRICS SAVED TO: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
