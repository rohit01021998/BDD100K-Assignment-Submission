"""
data_loader.py — Parser & Data Structure for BDD100K / COCO-style Object Detection Annotations.

Parses BDD100K JSON label files for train/val splits, filters to bounding-box
(box2d) annotations, converts coordinates from (x1, y1, x2, y2) to (x, y, w, h),
and flattens everything into a single Pandas DataFrame for downstream analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class COCODatasetParser:
    """Parses BDD100K-formatted JSON annotations and produces a flat DataFrame.

    The BDD100K format stores each image as a dict with:
        - ``name``: filename
        - ``labels``: list of annotation dicts, each with ``category``,
          ``box2d`` (x1, y1, x2, y2), and ``id``

    Labels without ``box2d`` (e.g., poly2d for lanes / drivable areas) are
    automatically filtered out.

    Attributes:
        df: The fully parsed, concatenated DataFrame across all splits.
    """

    # Columns produced by the parser
    COLUMNS: List[str] = [
        "image_id",
        "file_name",
        "split",
        "category_name",
        "annotation_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "area",
        "aspect_ratio",
    ]

    def __init__(self) -> None:
        self._dataframes: List[pd.DataFrame] = []
        self.df: pd.DataFrame = pd.DataFrame(columns=self.COLUMNS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        annotation_paths: Dict[str, str],
    ) -> pd.DataFrame:
        """Parse one or more annotation files and merge into a single DataFrame.

        Args:
            annotation_paths: Mapping of ``{split_name: json_file_path}``,
                e.g. ``{"train": "/path/to/train.json", "val": "/path/to/val.json"}``.

        Returns:
            A DataFrame with one row per bounding-box annotation, containing
            the columns listed in :pyattr:`COLUMNS`.
        """
        self._dataframes.clear()

        for split_name, json_path in annotation_paths.items():
            logger.info("Parsing %s split from %s …", split_name, json_path)
            split_df = self._parse_single_split(json_path, split_name)
            self._dataframes.append(split_df)
            logger.info(
                "  → %s split: %d images, %d box annotations",
                split_name,
                split_df["image_id"].nunique(),
                len(split_df),
            )

        self.df = pd.concat(self._dataframes, ignore_index=True)
        logger.info(
            "Total dataset: %d images, %d annotations, %d classes",
            self.df["image_id"].nunique(),
            len(self.df),
            self.df["category_name"].nunique(),
        )
        return self.df

    def get_split(self, split: str) -> pd.DataFrame:
        """Return the subset of the DataFrame for a given split.

        Args:
            split: One of the split names passed during :pymeth:`parse`.

        Returns:
            Filtered DataFrame.
        """
        return self.df[self.df["split"] == split].copy()

    def get_classes(self) -> List[str]:
        """Return a sorted list of unique class names."""
        return sorted(self.df["category_name"].unique().tolist())

    def summary(self) -> Dict[str, object]:
        """Return a high-level summary dict of the parsed dataset."""
        return {
            "total_images": int(self.df["image_id"].nunique()),
            "total_annotations": len(self.df),
            "classes": self.get_classes(),
            "splits": {
                split: {
                    "images": int(sub["image_id"].nunique()),
                    "annotations": len(sub),
                }
                for split, sub in self.df.groupby("split")
            },
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_single_split(
        self,
        json_path: str,
        split_name: str,
    ) -> pd.DataFrame:
        """Load a single JSON file and flatten to rows.

        Args:
            json_path: Absolute path to the BDD100K label JSON.
            split_name: Label to store in the ``split`` column.

        Returns:
            DataFrame for this split.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw_data: List[Dict] = json.load(fh)

        rows: List[Dict] = []
        for img_idx, image_entry in enumerate(raw_data):
            file_name: str = image_entry.get("name", "")
            image_id: str = f"{split_name}_{img_idx}"

            labels: List[Dict] = image_entry.get("labels", [])
            for label in labels:
                box2d: Optional[Dict] = label.get("box2d")
                if box2d is None:
                    # Skip non-box annotations (poly2d, etc.)
                    continue

                category: str = label.get("category", "unknown")
                annotation_id: int = label.get("id", -1)

                # Convert (x1, y1, x2, y2) → (x, y, w, h)
                x1 = float(box2d["x1"])
                y1 = float(box2d["y1"])
                x2 = float(box2d["x2"])
                y2 = float(box2d["y2"])

                bbox_w = max(x2 - x1, 0.0)
                bbox_h = max(y2 - y1, 0.0)
                area = bbox_w * bbox_h
                aspect_ratio = (bbox_w / bbox_h) if bbox_h > 0 else 0.0

                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "split": split_name,
                        "category_name": category,
                        "annotation_id": annotation_id,
                        "bbox_x": x1,
                        "bbox_y": y1,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                    }
                )

        return pd.DataFrame(rows, columns=self.COLUMNS)
