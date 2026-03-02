"""
outlier_detector.py — Per-Class Anomaly Analysis for Object Detection.

Uses both IQR-based statistical methods and Isolation Forest (sklearn) to
detect anomalous bounding boxes within each object class independently.
Incorporates both geometric features (area, aspect_ratio, width, height)
AND positional features (center_x, center_y) for anomaly detection.

Identifies:
  - Absolute largest / smallest objects per class
  - Most extreme aspect ratios (tall/skinny, wide/flat)
  - Positional outliers (objects in unusual image locations for their class)
  - Multi-feature statistical outliers (IQR + Isolation Forest)
  - Most crowded image per class
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Assumed image dimensions for BDD100K (1280×720) — used for normalisation
IMAGE_W = 1280
IMAGE_H = 720


@dataclass
class FlaggedSample:
    """A single flagged bounding-box annotation with context.

    Attributes:
        image_id: Internal image identifier.
        file_name: Original image filename.
        split: Dataset split (train / val).
        annotation_id: Annotation identifier within the image.
        category_name: Object class name.
        bbox: Tuple of (x, y, w, h).
        area: Area of the bounding box.
        aspect_ratio: Width-to-height ratio.
        center_x: Bounding-box center X coordinate.
        center_y: Bounding-box center Y coordinate.
        reason: Human-readable reason for flagging.
    """

    image_id: str
    file_name: str
    split: str
    annotation_id: int
    category_name: str
    bbox: tuple  # (x, y, w, h)
    area: float
    aspect_ratio: float
    center_x: float
    center_y: float
    reason: str


@dataclass
class ClassAnomalyReport:
    """Aggregated anomaly report for a single class.

    Attributes:
        class_name: The object class.
        total_annotations: Count of annotations for this class.
        largest: Flagged samples — largest objects.
        smallest: Flagged samples — smallest objects.
        tallest: Flagged samples — most extreme tall/skinny aspect ratios.
        widest: Flagged samples — most extreme wide/flat aspect ratios.
        position_outliers: Flagged samples — objects in unusual locations.
        statistical_outliers: Samples flagged by IQR or Isolation Forest.
        most_crowded_image: All annotations from the most crowded image.
    """

    class_name: str
    total_annotations: int = 0
    largest: List[FlaggedSample] = field(default_factory=list)
    smallest: List[FlaggedSample] = field(default_factory=list)
    tallest: List[FlaggedSample] = field(default_factory=list)
    widest: List[FlaggedSample] = field(default_factory=list)
    position_outliers: List[FlaggedSample] = field(default_factory=list)
    statistical_outliers: List[FlaggedSample] = field(default_factory=list)
    most_crowded_image: List[FlaggedSample] = field(default_factory=list)


class AnomalyDetector:
    """Per-class anomaly detection on bounding-box geometry AND position.

    Applies both IQR-based and Isolation-Forest-based detection
    independently for each object class, using:
      - Geometric features: area, aspect_ratio, bbox_w, bbox_h
      - Positional features: center_x, center_y (normalised to [0, 1])

    Args:
        df: Flat annotations DataFrame produced by :class:`COCODatasetParser`.
        top_k: Number of extreme samples to keep per category per flag type.
        iqr_multiplier: IQR fence multiplier (default 1.5 for mild outliers).
        contamination: Isolation Forest contamination parameter.
    """

    GEOMETRIC_FEATURES = ["area", "aspect_ratio", "bbox_w", "bbox_h"]
    POSITION_FEATURES = ["center_x_norm", "center_y_norm"]
    ALL_FEATURES = GEOMETRIC_FEATURES + POSITION_FEATURES

    def __init__(
        self,
        df: pd.DataFrame,
        top_k: int = 3,
        iqr_multiplier: float = 1.5,
        contamination: float = 0.05,
    ) -> None:
        self.df = df.copy()
        self.top_k = top_k
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination
        self.reports: Dict[str, ClassAnomalyReport] = {}

        # Pre-compute center coordinates
        self.df["center_x"] = self.df["bbox_x"] + self.df["bbox_w"] / 2
        self.df["center_y"] = self.df["bbox_y"] + self.df["bbox_h"] / 2
        # Normalised to [0, 1] for Isolation Forest
        self.df["center_x_norm"] = self.df["center_x"] / IMAGE_W
        self.df["center_y_norm"] = self.df["center_y"] / IMAGE_H

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, ClassAnomalyReport]:
        """Run anomaly detection for every class.

        Returns:
            Dict mapping class_name → ClassAnomalyReport.
        """
        classes = sorted(self.df["category_name"].unique())
        logger.info(
            "Running per-class anomaly detection for %d classes …", len(classes)
        )

        for cls_name in classes:
            cls_df = self.df[self.df["category_name"] == cls_name].copy()
            if cls_df.empty:
                continue
            report = self._analyze_class(cls_name, cls_df)
            self.reports[cls_name] = report
            n_flags = (
                len(report.largest)
                + len(report.smallest)
                + len(report.tallest)
                + len(report.widest)
                + len(report.position_outliers)
                + len(report.statistical_outliers)
                + len(report.most_crowded_image)
            )
            logger.info(
                "  %s: %d annotations → %d flagged samples",
                cls_name,
                report.total_annotations,
                n_flags,
            )

        logger.info("Anomaly detection complete for %d classes.", len(self.reports))
        return self.reports

    def get_all_flagged_samples(self) -> List[FlaggedSample]:
        """Return a flat list of all flagged samples across all classes."""
        samples: List[FlaggedSample] = []
        for report in self.reports.values():
            samples.extend(report.largest)
            samples.extend(report.smallest)
            samples.extend(report.tallest)
            samples.extend(report.widest)
            samples.extend(report.position_outliers)
            samples.extend(report.statistical_outliers)
            samples.extend(report.most_crowded_image)
        return samples

    # ------------------------------------------------------------------
    # Per-class analysis
    # ------------------------------------------------------------------

    def _analyze_class(
        self, cls_name: str, cls_df: pd.DataFrame
    ) -> ClassAnomalyReport:
        """Run all anomaly detectors on a single class's annotations.

        Args:
            cls_name: Class label.
            cls_df: Subset DataFrame for this class.

        Returns:
            Populated ClassAnomalyReport.
        """
        report = ClassAnomalyReport(
            class_name=cls_name,
            total_annotations=len(cls_df),
        )

        # --- Extreme sizes ---
        report.largest = self._top_by_column(
            cls_df, "area", ascending=False, reason=f"LARGEST {cls_name}"
        )
        report.smallest = self._top_by_column(
            cls_df, "area", ascending=True, reason=f"SMALLEST {cls_name}"
        )

        # --- Extreme aspect ratios ---
        report.tallest = self._top_by_column(
            cls_df,
            "aspect_ratio",
            ascending=True,
            reason=f"TALLEST (skinny) {cls_name}",
        )
        report.widest = self._top_by_column(
            cls_df,
            "aspect_ratio",
            ascending=False,
            reason=f"WIDEST (flat) {cls_name}",
        )

        # --- Position outliers ---
        report.position_outliers = self._detect_position_outliers(cls_df, cls_name)

        # --- Statistical outliers (IQR + Isolation Forest on ALL features) ---
        report.statistical_outliers = self._detect_statistical_outliers(
            cls_df, cls_name
        )

        # --- Most crowded image ---
        report.most_crowded_image = self._most_crowded_image(cls_df, cls_name)

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_flagged(self, row: pd.Series, reason: str) -> FlaggedSample:
        """Convert a DataFrame row to a FlaggedSample."""
        return FlaggedSample(
            image_id=str(row["image_id"]),
            file_name=str(row["file_name"]),
            split=str(row["split"]),
            annotation_id=int(row["annotation_id"]),
            category_name=str(row["category_name"]),
            bbox=(
                float(row["bbox_x"]),
                float(row["bbox_y"]),
                float(row["bbox_w"]),
                float(row["bbox_h"]),
            ),
            area=float(row["area"]),
            aspect_ratio=float(row["aspect_ratio"]),
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            reason=reason,
        )

    def _top_by_column(
        self,
        cls_df: pd.DataFrame,
        column: str,
        ascending: bool,
        reason: str,
    ) -> List[FlaggedSample]:
        """Return top-k rows sorted by a column.

        Args:
            cls_df: Class-filtered DataFrame.
            column: Column to sort by.
            ascending: Sort direction.
            reason: Reason string for the flag.

        Returns:
            List of FlaggedSample.
        """
        sorted_df = cls_df.sort_values(column, ascending=ascending).head(self.top_k)
        return [self._row_to_flagged(row, reason) for _, row in sorted_df.iterrows()]

    def _detect_position_outliers(
        self, cls_df: pd.DataFrame, cls_name: str
    ) -> List[FlaggedSample]:
        """Detect objects at unusual positions for their class using IQR on center_x/center_y.

        For each class, we compute the typical position envelope and flag objects
        that fall outside the IQR fences for either center_x or center_y.

        Args:
            cls_df: Class-filtered DataFrame.
            cls_name: Class name for labelling.

        Returns:
            List of flagged position-anomalous samples.
        """
        outlier_mask = pd.Series(False, index=cls_df.index)

        for col in ["center_x", "center_y"]:
            q1 = cls_df[col].quantile(0.05)
            q3 = cls_df[col].quantile(0.95)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            outlier_mask = outlier_mask | (cls_df[col] < lower) | (cls_df[col] > upper)

        flagged: List[FlaggedSample] = []
        outlier_df = cls_df[outlier_mask]
        if outlier_df.empty:
            return flagged

        # Sort by distance from class centroid to get the most extreme
        cx_mean = cls_df["center_x"].mean()
        cy_mean = cls_df["center_y"].mean()
        distances = np.sqrt(
            (outlier_df["center_x"] - cx_mean) ** 2
            + (outlier_df["center_y"] - cy_mean) ** 2
        )
        top_indices = distances.nlargest(self.top_k).index

        for idx in top_indices:
            row = cls_df.loc[idx]
            cx, cy = row["center_x"], row["center_y"]
            flagged.append(
                self._row_to_flagged(
                    row,
                    f"POSITION outlier {cls_name} (cx={cx:.0f}, cy={cy:.0f})",
                )
            )
        return flagged

    def _detect_statistical_outliers(
        self, cls_df: pd.DataFrame, cls_name: str
    ) -> List[FlaggedSample]:
        """Detect outliers via IQR method and Isolation Forest.

        Uses BOTH geometric features (area, aspect_ratio, bbox_w, bbox_h)
        AND positional features (center_x_norm, center_y_norm) so that
        an object can be flagged for being anomalous in size, shape, or position.

        Args:
            cls_df: Class-filtered DataFrame.
            cls_name: Class name for labelling.

        Returns:
            De-duplicated list of flagged outlier samples.
        """
        outlier_indices: set = set()

        # ---- IQR method on area and aspect_ratio ----
        for col in ["area", "aspect_ratio"]:
            q1 = cls_df[col].quantile(0.25)
            q3 = cls_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            mask = (cls_df[col] < lower) | (cls_df[col] > upper)
            outlier_indices.update(cls_df[mask].index.tolist())

        # ---- Isolation Forest on ALL features (geometry + position) ----
        if len(cls_df) >= 10:
            try:
                features = cls_df[self.ALL_FEATURES].values
                iso = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_jobs=-1,
                )
                preds = iso.fit_predict(features)
                iso_outliers = cls_df.index[preds == -1].tolist()
                outlier_indices.update(iso_outliers)
            except Exception as exc:
                logger.warning(
                    "Isolation Forest failed for class '%s': %s", cls_name, exc
                )

        # Build flagged samples (cap to avoid overwhelming output)
        flagged: List[FlaggedSample] = []
        for idx in list(outlier_indices)[: self.top_k * 3]:
            row = cls_df.loc[idx]
            flagged.append(self._row_to_flagged(row, f"OUTLIER {cls_name}"))
        return flagged

    def _most_crowded_image(
        self, cls_df: pd.DataFrame, cls_name: str
    ) -> List[FlaggedSample]:
        """Find the image with the most annotations for this class.

        Args:
            cls_df: Class-filtered DataFrame.
            cls_name: Class name for labelling.

        Returns:
            List of all annotations from the most crowded image.
        """
        counts = cls_df.groupby("image_id").size()
        most_crowded_id = counts.idxmax()
        crowded_df = cls_df[cls_df["image_id"] == most_crowded_id]
        reason = f"CROWDED ({len(crowded_df)} {cls_name} instances in one image)"
        return [
            self._row_to_flagged(row, reason) for _, row in crowded_df.iterrows()
        ]
