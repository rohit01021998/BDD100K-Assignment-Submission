"""
analyzer.py — Distribution & Split Analysis for Object Detection Datasets.

Generates publication-quality visualizations including:
  - Global class distribution (train vs val)
  - Per-class dedicated analysis pages (area distribution, aspect ratio, position heatmap)
  - Detections-per-image CDF
  - Box area & aspect ratio KDE overlays
  - Per-class violin plots
  - Class co-occurrence heatmap
  - Class-wise spatial heatmaps (where objects appear in the image)

Every plot includes an engineer-facing annotation box explaining the method,
what it shows, and how to interpret the results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global plot style (Catppuccin Mocha inspired)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.facecolor": "#1e1e2e",
        "axes.facecolor": "#1e1e2e",
        "axes.edgecolor": "#cdd6f4",
        "axes.labelcolor": "#cdd6f4",
        "text.color": "#cdd6f4",
        "xtick.color": "#cdd6f4",
        "ytick.color": "#cdd6f4",
        "legend.facecolor": "#313244",
        "legend.edgecolor": "#585b70",
        "font.family": "sans-serif",
        "font.size": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    }
)

# Catppuccin-inspired palette
PALETTE_TRAIN = "#89b4fa"  # blue
PALETTE_VAL = "#f38ba8"  # pink
PALETTE_ACCENT = "#a6e3a1"  # green

# Assumed image dimensions for BDD100K (1280×720)
IMAGE_W = 1280
IMAGE_H = 720


def _add_explanation(
    fig: plt.Figure,
    text: str,
    y: float = -0.02,
    fontsize: int = 9,
) -> None:
    """Add an engineer-facing explanation text box at the bottom of a figure.

    Args:
        fig: Matplotlib Figure.
        text: Multi-line explanation string.
        y: Vertical position (in figure coords, < 0 = below axes).
        fontsize: Font size for the note.
    """
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="#a6adc8",
        fontstyle="italic",
        wrap=True,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#313244",
            edgecolor="#585b70",
            alpha=0.9,
        ),
    )


class DistributionAnalyzer:
    """Generates and saves distribution / split-comparison plots.

    All plots are written to ``output_dir`` as high-resolution PNGs.
    Each plot includes a brief explanation at the bottom describing:
      - **What** the plot shows
      - **Method** used to generate it
      - **How to infer** actionable insights from it

    Args:
        df: The flat annotations DataFrame produced by :class:`COCODatasetParser`.
        output_dir: Directory where plot images will be saved.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "output/plots") -> None:
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._per_class_dir = self.output_dir / "per_class"
        self._per_class_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API — run all analyses
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        """Execute every analysis method and save all plots."""
        logger.info("Running distribution analysis …")
        self.plot_class_distribution()
        self.plot_detections_per_image_cdf()
        self.plot_area_kde()
        self.plot_aspect_ratio_kde()
        self.plot_per_class_area_violin()
        self.plot_co_occurrence_heatmap()
        self.plot_global_spatial_heatmap()
        self.plot_per_class_analysis()
        logger.info("All distribution plots saved to %s", self.output_dir)

    # ------------------------------------------------------------------
    # 1. Class Distribution
    # ------------------------------------------------------------------

    def plot_class_distribution(self) -> Path:
        """Horizontal bar chart of annotation counts per class, train vs val.

        Returns:
            Path to the saved plot.
        """
        class_counts = (
            self.df.groupby(["category_name", "split"]).size().unstack(fill_value=0)
        )
        class_counts["_total"] = class_counts.sum(axis=1)
        class_counts = class_counts.sort_values("_total", ascending=True)
        class_counts = class_counts.drop(columns="_total")

        fig, ax = plt.subplots(figsize=(12, max(6, len(class_counts) * 0.6)))

        y_pos = np.arange(len(class_counts))
        bar_height = 0.35
        splits = [c for c in ["train", "val"] if c in class_counts.columns]
        colors = {"train": PALETTE_TRAIN, "val": PALETTE_VAL}

        for i, split in enumerate(splits):
            offset = (i - 0.5) * bar_height
            ax.barh(
                y_pos + offset,
                class_counts[split],
                height=bar_height,
                label=split.capitalize(),
                color=colors.get(split, PALETTE_ACCENT),
                edgecolor="none",
                alpha=0.9,
            )
            for j, val in enumerate(class_counts[split]):
                if val > 0:
                    ax.text(
                        val + class_counts.values.max() * 0.01,
                        y_pos[j] + offset,
                        f"{val:,}",
                        va="center",
                        fontsize=8,
                        color=colors.get(split, "#cdd6f4"),
                    )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_counts.index, fontsize=10)
        ax.set_xlabel("Number of Annotations")
        ax.set_title(
            "Class Distribution — Train vs Val", fontsize=16, fontweight="bold"
        )
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.15, color="#cdd6f4")

        _add_explanation(
            fig,
            "WHAT: Shows annotation count per object class for each split. "
            "METHOD: Simple frequency count grouped by class and split. "
            "INFERENCE: Check for class imbalance — if minority classes (e.g., 'train', 'rider') have "
            "orders-of-magnitude fewer samples than 'car', the model may need oversampling, "
            "focal loss, or class-weighted training. Also verify train/val ratio is consistent across classes.",
        )

        path = self.output_dir / "class_distribution.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 2. Detections per Image CDF
    # ------------------------------------------------------------------

    def plot_detections_per_image_cdf(self) -> Path:
        """CDF of detections per image, comparing train vs val.

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = self.df[self.df["split"] == split]
            if sub.empty:
                continue
            counts = sub.groupby("image_id").size().values
            sorted_counts = np.sort(counts)
            cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            ax.plot(
                sorted_counts,
                cdf,
                label=f"{split.capitalize()} (median={int(np.median(counts))})",
                color=color,
                linewidth=2,
            )

        ax.set_xlabel("Detections per Image")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(
            "CDF — Detections per Image (Train vs Val)",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(alpha=0.15, color="#cdd6f4")

        _add_explanation(
            fig,
            "WHAT: Empirical CDF of the number of detected objects per image. "
            "METHOD: For each image, count annotations; sort and plot cumulative fraction. "
            "INFERENCE: If train and val curves diverge, the splits have different scene complexity. "
            "A long tail (CDF rising slowly) means some images are very crowded — consider "
            "NMS tuning and anchor density. The median value indicates typical scene density.",
        )

        path = self.output_dir / "detections_per_image_cdf.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 3. Box Area KDE
    # ------------------------------------------------------------------

    def plot_area_kde(self) -> Path:
        """KDE of bounding-box area (log-scale) overlaid for train vs val.

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = self.df[self.df["split"] == split]
            if sub.empty:
                continue
            log_area = np.log10(sub["area"].clip(lower=1))
            ax.hist(log_area, bins=120, density=True, alpha=0.25, color=color)
            try:
                sns.kdeplot(
                    log_area,
                    ax=ax,
                    color=color,
                    linewidth=2,
                    label=f"{split.capitalize()} KDE",
                    warn_singular=False,
                )
            except Exception:
                pass

        ax.set_xlabel("log₁₀(Box Area in px²)")
        ax.set_ylabel("Density")
        ax.set_title(
            "Box Area Distribution — Train vs Val", fontsize=16, fontweight="bold"
        )
        ax.legend()
        ax.grid(alpha=0.15, color="#cdd6f4")

        _add_explanation(
            fig,
            "WHAT: Probability density of bounding-box areas on a log₁₀ scale. "
            "METHOD: Kernel Density Estimation (Gaussian kernel, Scott bandwidth) on log-transformed area. "
            "INFERENCE: If train/val peaks misalign, there is a size distribution shift between splits — "
            "could bias anchor generation. Multi-modal peaks indicate distinct object-size clusters "
            "(e.g., distant vs. nearby objects). Extremely small areas (< 10² px²) may be labeling noise.",
        )

        path = self.output_dir / "area_kde.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 4. Aspect Ratio KDE
    # ------------------------------------------------------------------

    def plot_aspect_ratio_kde(self) -> Path:
        """KDE of aspect ratio overlaid for train vs val.

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = self.df[self.df["split"] == split]
            if sub.empty:
                continue
            ar = sub["aspect_ratio"].clip(upper=10)
            ax.hist(ar, bins=120, density=True, alpha=0.25, color=color)
            try:
                sns.kdeplot(
                    ar,
                    ax=ax,
                    color=color,
                    linewidth=2,
                    label=f"{split.capitalize()} KDE",
                    warn_singular=False,
                )
            except Exception:
                pass

        ax.set_xlabel("Aspect Ratio (w / h)")
        ax.set_ylabel("Density")
        ax.set_title(
            "Aspect Ratio Distribution — Train vs Val",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(alpha=0.15, color="#cdd6f4")

        _add_explanation(
            fig,
            "WHAT: Density of bounding-box aspect ratios (width ÷ height). "
            "METHOD: KDE on clipped aspect ratios (capped at 10 to remove extreme outliers). "
            "INFERENCE: AR ≈ 1 means roughly square boxes. Peaks at specific ARs reveal class-specific "
            "shapes (e.g., traffic lights tend to be tall/narrow → AR < 1, cars tend to be wide → AR > 1). "
            "Use this to inform anchor-box aspect ratio choices in detectors like Faster R-CNN or SSD.",
        )

        path = self.output_dir / "aspect_ratio_kde.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 5. Per-Class Area Violin
    # ------------------------------------------------------------------

    def plot_per_class_area_violin(self) -> Path:
        """Violin plots of log-area per class.

        Returns:
            Path to the saved plot.
        """
        df_plot = self.df.copy()
        df_plot["log_area"] = np.log10(df_plot["area"].clip(lower=1))

        order = (
            df_plot.groupby("category_name")["log_area"]
            .median()
            .sort_values()
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(14, max(6, len(order) * 0.7)))
        sns.violinplot(
            data=df_plot,
            y="category_name",
            x="log_area",
            order=order,
            hue="split",
            split=True if df_plot["split"].nunique() == 2 else False,
            inner="quartile",
            palette={"train": PALETTE_TRAIN, "val": PALETTE_VAL},
            ax=ax,
            linewidth=0.8,
            density_norm="width",
        )
        ax.set_xlabel("log₁₀(Box Area in px²)")
        ax.set_ylabel("")
        ax.set_title(
            "Per-Class Box Area Violin — Train vs Val",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.15, color="#cdd6f4")

        _add_explanation(
            fig,
            "WHAT: Full area distribution shape for each class, with train (blue) and val (pink) side-by-side. "
            "METHOD: Violin plot on log₁₀-transformed area; internal lines show quartiles (Q1, median, Q3). "
            "INFERENCE: Compare shape symmetry — a long left tail means many very small instances "
            "(potentially hard to detect). If train/val violin shapes differ for a class, "
            "expect domain shift in evaluation. Wide violins = high variance in object size.",
        )

        path = self.output_dir / "per_class_area_violin.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 6. Co-occurrence Heatmap
    # ------------------------------------------------------------------

    def plot_co_occurrence_heatmap(self) -> Path:
        """Class co-occurrence heatmap — which classes appear together.

        Returns:
            Path to the saved plot.
        """
        classes = sorted(self.df["category_name"].unique())
        n = len(classes)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        co_matrix = np.zeros((n, n), dtype=np.int64)

        for _, group in self.df.groupby("image_id"):
            cats = group["category_name"].unique()
            for i, c1 in enumerate(cats):
                for c2 in cats[i:]:
                    idx1, idx2 = cls_to_idx[c1], cls_to_idx[c2]
                    co_matrix[idx1, idx2] += 1
                    if idx1 != idx2:
                        co_matrix[idx2, idx1] += 1

        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)))
        sns.heatmap(
            co_matrix,
            xticklabels=classes,
            yticklabels=classes,
            cmap="magma",
            annot=True if n <= 15 else False,
            fmt="d" if n <= 15 else "",
            linewidths=0.5,
            linecolor="#313244",
            ax=ax,
            cbar_kws={"label": "Co-occurrence Count"},
        )
        ax.set_title(
            "Class Co-Occurrence Heatmap", fontsize=16, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        _add_explanation(
            fig,
            "WHAT: How often each pair of classes co-occurs in the same image. "
            "Diagonal = number of images containing that class. "
            "METHOD: Count unique image-level co-occurrences for all class pairs. "
            "INFERENCE: High off-diagonal values indicate contextual relationships "
            "(e.g., 'rider' ↔ 'motor'). Use this to design multi-label heads, "
            "contextual reasoning modules, or to validate that rare class samples "
            "include realistic surrounding context.",
        )

        path = self.output_dir / "co_occurrence_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 7. Global Spatial Heatmap
    # ------------------------------------------------------------------

    def plot_global_spatial_heatmap(self) -> Path:
        """Combined spatial heatmap of all bounding-box centers.

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        cx = self.df["bbox_x"] + self.df["bbox_w"] / 2
        cy = self.df["bbox_y"] + self.df["bbox_h"] / 2

        heatmap, xedges, yedges = np.histogram2d(
            cx.values, cy.values, bins=[64, 36], range=[[0, IMAGE_W], [0, IMAGE_H]]
        )
        ax.imshow(
            heatmap.T,
            origin="upper",
            extent=[0, IMAGE_W, IMAGE_H, 0],
            cmap="inferno",
            aspect="auto",
        )
        ax.set_xlabel("Image X (px)")
        ax.set_ylabel("Image Y (px)")
        ax.set_title(
            "Global Spatial Heatmap — All Object Centers",
            fontsize=16,
            fontweight="bold",
        )

        _add_explanation(
            fig,
            "WHAT: Spatial density of bounding-box centers across all classes and images. "
            "METHOD: 2D histogram (64×36 bins) of box center coordinates. "
            "INFERENCE: High-density regions show where objects are most commonly annotated. "
            "Expect a 'horizon band' pattern for driving datasets. Dead zones (corners) suggest "
            "the detector may see few training examples there — consider augmentation strategies "
            "like random cropping or mosaic augmentation.",
        )

        path = self.output_dir / "global_spatial_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # 8. Per-Class Dedicated Analysis (multi-panel per class)
    # ------------------------------------------------------------------

    def plot_per_class_analysis(self) -> None:
        """Generate a dedicated multi-panel analysis page for each class.

        For each class, produces a 2×2 figure containing:
          - Area distribution histogram + KDE
          - Aspect ratio distribution histogram + KDE
          - Spatial heatmap (where this class appears in the image)
          - Box center scatter plot (position distribution)
        """
        classes = sorted(self.df["category_name"].unique())
        logger.info(
            "  Generating per-class analysis for %d classes …", len(classes)
        )

        for cls_name in classes:
            cls_df = self.df[self.df["category_name"] == cls_name]
            self._plot_single_class_analysis(cls_name, cls_df)

    def _plot_single_class_analysis(
        self, cls_name: str, cls_df: pd.DataFrame
    ) -> Path:
        """Generate a 2×2 multi-panel analysis page for a single class.

        Args:
            cls_name: Object class name.
            cls_df: Filtered DataFrame for this class.

        Returns:
            Path to the saved plot.
        """
        safe_name = cls_name.replace(" ", "_")
        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(
            f"Per-Class Analysis — {cls_name.title()}  "
            f"({len(cls_df):,} annotations)",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3, top=0.93, bottom=0.12)

        # --- Panel 1: Area Distribution (train vs val) ---
        ax1 = fig.add_subplot(gs[0, 0])
        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = cls_df[cls_df["split"] == split]
            if sub.empty:
                continue
            log_area = np.log10(sub["area"].clip(lower=1))
            ax1.hist(log_area, bins=60, density=True, alpha=0.3, color=color)
            try:
                sns.kdeplot(
                    log_area,
                    ax=ax1,
                    color=color,
                    linewidth=2,
                    label=f"{split} (n={len(sub):,})",
                    warn_singular=False,
                )
            except Exception:
                pass
        ax1.set_xlabel("log₁₀(Area in px²)")
        ax1.set_ylabel("Density")
        ax1.set_title(f"Area Distribution — {cls_name}", fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.15, color="#cdd6f4")

        # --- Panel 2: Aspect Ratio Distribution (train vs val) ---
        ax2 = fig.add_subplot(gs[0, 1])
        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = cls_df[cls_df["split"] == split]
            if sub.empty:
                continue
            ar = sub["aspect_ratio"].clip(upper=8)
            ax2.hist(ar, bins=60, density=True, alpha=0.3, color=color)
            try:
                sns.kdeplot(
                    ar,
                    ax=ax2,
                    color=color,
                    linewidth=2,
                    label=f"{split} (n={len(sub):,})",
                    warn_singular=False,
                )
            except Exception:
                pass
        ax2.set_xlabel("Aspect Ratio (w / h)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Aspect Ratio Distribution — {cls_name}", fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.15, color="#cdd6f4")

        # --- Panel 3: Spatial Heatmap ---
        ax3 = fig.add_subplot(gs[1, 0])
        cx = cls_df["bbox_x"] + cls_df["bbox_w"] / 2
        cy = cls_df["bbox_y"] + cls_df["bbox_h"] / 2
        heatmap, _, _ = np.histogram2d(
            cx.values, cy.values, bins=[64, 36], range=[[0, IMAGE_W], [0, IMAGE_H]]
        )
        im = ax3.imshow(
            heatmap.T,
            origin="upper",
            extent=[0, IMAGE_W, IMAGE_H, 0],
            cmap="inferno",
            aspect="auto",
        )
        fig.colorbar(im, ax=ax3, label="Count", shrink=0.8)
        ax3.set_xlabel("Image X (px)")
        ax3.set_ylabel("Image Y (px)")
        ax3.set_title(f"Spatial Heatmap — {cls_name}", fontsize=13)

        # --- Panel 4: Box Center Scatter (position + area encoding) ---
        ax4 = fig.add_subplot(gs[1, 1])
        # Subsample if too many points
        plot_df = cls_df if len(cls_df) <= 5000 else cls_df.sample(5000, random_state=42)
        cx_s = plot_df["bbox_x"] + plot_df["bbox_w"] / 2
        cy_s = plot_df["bbox_y"] + plot_df["bbox_h"] / 2
        sizes = np.clip(np.log10(plot_df["area"].clip(lower=1)) * 3, 2, 30)
        scatter = ax4.scatter(
            cx_s,
            cy_s,
            s=sizes,
            c=np.log10(plot_df["area"].clip(lower=1)),
            cmap="cool",
            alpha=0.4,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax4, label="log₁₀(Area)", shrink=0.8)
        ax4.set_xlim(0, IMAGE_W)
        ax4.set_ylim(IMAGE_H, 0)
        ax4.set_xlabel("Image X (px)")
        ax4.set_ylabel("Image Y (px)")
        ax4.set_title(f"Position vs Size — {cls_name}", fontsize=13)

        # --- Panel 5: Box Width vs Height scatter ---
        ax5 = fig.add_subplot(gs[2, 0])
        plot_df2 = cls_df if len(cls_df) <= 5000 else cls_df.sample(5000, random_state=42)
        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = plot_df2[plot_df2["split"] == split]
            if sub.empty:
                continue
            ax5.scatter(
                sub["bbox_w"],
                sub["bbox_h"],
                s=4,
                alpha=0.3,
                color=color,
                label=split,
                edgecolors="none",
            )
        ax5.set_xlabel("Box Width (px)")
        ax5.set_ylabel("Box Height (px)")
        ax5.set_title(f"Width vs Height Scatter — {cls_name}", fontsize=13)
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.15, color="#cdd6f4")

        # --- Panel 6: Detections per image for this class ---
        ax6 = fig.add_subplot(gs[2, 1])
        for split, color in [("train", PALETTE_TRAIN), ("val", PALETTE_VAL)]:
            sub = cls_df[cls_df["split"] == split]
            if sub.empty:
                continue
            counts = sub.groupby("image_id").size().values
            ax6.hist(
                counts,
                bins=min(max(counts), 50),
                density=True,
                alpha=0.5,
                color=color,
                label=f"{split} (max={max(counts)})",
                edgecolor="none",
            )
        ax6.set_xlabel(f"Number of '{cls_name}' per Image")
        ax6.set_ylabel("Density")
        ax6.set_title(f"Instance Count per Image — {cls_name}", fontsize=13)
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.15, color="#cdd6f4")

        # --- Explanation text ---
        _add_explanation(
            fig,
            f"PER-CLASS ANALYSIS: {cls_name.upper()}\n"
            f"• Area Distribution: KDE on log₁₀(area). Multi-modal peaks → distinct size clusters. "
            f"Train/val mismatch → potential domain shift.\n"
            f"• Aspect Ratio: KDE of w/h. Informs anchor box design; consistent peaks expected across splits.\n"
            f"• Spatial Heatmap: 2D histogram of box centers (64×36 bins). Shows where this class appears. "
            f"E.g., 'car' concentrates near horizon; 'traffic light' in upper regions.\n"
            f"• Position vs Size: Scatter of center position colored by area. Reveals if object size "
            f"correlates with vertical position (perspective effect).\n"
            f"• Width vs Height: Reveals the aspect-ratio cluster shape and dimensional range.\n"
            f"• Instance Count: How many instances of this class appear per image — useful for NMS tuning.",
            y=0.0,
            fontsize=8,
        )

        path = self._per_class_dir / f"{safe_name}_analysis.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("    Saved per-class analysis: %s", path.name)
        return path
