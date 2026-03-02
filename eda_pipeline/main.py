"""
main.py — Pipeline Orchestration for Object Detection EDA.

Ties together all pipeline components:
  1. Parse BDD100K annotations → DataFrame
  2. Generate distribution & split analysis plots
  3. Detect per-class anomalies (IQR + Isolation Forest)
  4. Render edge-case bounding boxes on original images
  5. Build a self-contained HTML dashboard

Usage:
    python -m eda_pipeline.main
"""

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these paths to match your setup
# ---------------------------------------------------------------------------

# Base directory of the BDD100K dataset (auto-detected from script location)
BASE_DIR = Path(__file__).resolve().parent.parent

# Annotation JSON files (one per split)
ANNOTATION_PATHS = {
    "train": str(
        BASE_DIR
        / "assignment_data_bdd"
        / "bdd100k_labels_release"
        / "bdd100k"
        / "labels"
        / "bdd100k_labels_images_train.json"
    ),
    "val": str(
        BASE_DIR
        / "assignment_data_bdd"
        / "bdd100k_labels_release"
        / "bdd100k"
        / "labels"
        / "bdd100k_labels_images_val.json"
    ),
}

# Image directories (one per split)
IMAGE_DIRS = {
    "train": str(
        BASE_DIR
        / "assignment_data_bdd"
        / "bdd100k_images_100k"
        / "bdd100k"
        / "images"
        / "100k"
        / "train"
    ),
    "val": str(
        BASE_DIR
        / "assignment_data_bdd"
        / "bdd100k_images_100k"
        / "bdd100k"
        / "images"
        / "100k"
        / "val"
    ),
}

# Output directory
OUTPUT_DIR = str(BASE_DIR / "eda_pipeline_output")

# Anomaly detection parameters
TOP_K_EXTREMES = 3
IQR_MULTIPLIER = 1.5
CONTAMINATION = 0.05

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("eda_pipeline")


def main() -> None:
    """Run the full EDA pipeline."""
    t0 = time.time()

    # --- Import pipeline components ---
    from .data_loader import COCODatasetParser
    from .analyzer import DistributionAnalyzer
    from .outlier_detector import AnomalyDetector
    from .visualizer import EdgeCaseVisualizer
    from .dashboard import DashboardBuilder

    # --- Step 1: Parse annotations ---
    logger.info("=" * 60)
    logger.info("STEP 1 / 5 — Parsing annotations")
    logger.info("=" * 60)
    parser = COCODatasetParser()
    df = parser.parse(ANNOTATION_PATHS)
    summary = parser.summary()

    logger.info("Dataset summary:")
    for key, val in summary.items():
        if key != "splits":
            logger.info("  %s: %s", key, val)
    for split_name, split_info in summary.get("splits", {}).items():
        logger.info("  %s: %s", split_name, split_info)

    # --- Step 2: Distribution analysis ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 / 5 — Distribution & split analysis")
    logger.info("=" * 60)
    plots_dir = str(Path(OUTPUT_DIR) / "plots")
    analyzer = DistributionAnalyzer(df, output_dir=plots_dir)
    analyzer.run_all()

    # --- Step 3: Anomaly detection ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 / 5 — Per-class anomaly detection")
    logger.info("=" * 60)
    detector = AnomalyDetector(
        df,
        top_k=TOP_K_EXTREMES,
        iqr_multiplier=IQR_MULTIPLIER,
        contamination=CONTAMINATION,
    )
    reports = detector.run()
    flagged = detector.get_all_flagged_samples()
    logger.info("Total flagged samples: %d", len(flagged))

    # --- Step 4: Edge-case visualization ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4 / 5 — Rendering edge-case bounding boxes")
    logger.info("=" * 60)
    edge_dir = str(Path(OUTPUT_DIR) / "edge_cases")
    visualizer = EdgeCaseVisualizer(
        image_dirs=IMAGE_DIRS,
        output_dir=edge_dir,
    )
    rendered_paths = visualizer.render_all(flagged)

    # --- Step 5: Build dashboard ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5 / 5 — Building HTML dashboard")
    logger.info("=" * 60)
    dashboard = DashboardBuilder(output_dir=OUTPUT_DIR)
    html_path = dashboard.build(summary=summary, edge_case_paths=rendered_paths)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅  Pipeline complete in %.1f s", elapsed)
    logger.info("   Dashboard: %s", html_path)
    logger.info("   Plots:     %s", plots_dir)
    logger.info("   Edge cases: %s (%d images)", edge_dir, len(rendered_paths))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
