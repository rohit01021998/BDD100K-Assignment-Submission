"""
visualizer.py — Image & Bounding Box Rendering for Edge Cases.

Loads original images, draws clearly visible bounding boxes with class labels
and anomaly reason text, and saves the rendered visualizations to disk.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .outlier_detector import FlaggedSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color map for anomaly types (BGR format for OpenCV)
# ---------------------------------------------------------------------------
REASON_COLORS: Dict[str, Tuple[int, int, int]] = {
    "LARGEST": (0, 200, 100),      # green
    "SMALLEST": (0, 180, 255),     # orange
    "TALLEST": (255, 100, 100),    # blue
    "WIDEST": (100, 100, 255),     # red
    "POSITION": (255, 0, 255),     # magenta
    "OUTLIER": (0, 0, 255),        # bright red
    "CROWDED": (255, 200, 0),      # cyan-ish
}

DEFAULT_COLOR: Tuple[int, int, int] = (200, 200, 200)  # grey fallback


def _color_for_reason(reason: str) -> Tuple[int, int, int]:
    """Pick a color based on the first keyword in the reason string."""
    reason_upper = reason.upper()
    for keyword, color in REASON_COLORS.items():
        if keyword in reason_upper:
            return color
    return DEFAULT_COLOR


class EdgeCaseVisualizer:
    """Draws bounding boxes on original images for flagged edge-case samples.

    For each flagged sample, the visualizer:
      1. Loads the original image from the dataset directory.
      2. Draws the bounding box with a colour-coded outline.
      3. Overlays the class label and anomaly reason.
      4. Saves the annotated image to ``output_dir``.

    Args:
        image_dirs: Mapping of ``{split_name: image_directory_path}``.
        output_dir: Directory to save rendered edge-case images.
        line_thickness: Bounding-box line thickness.
        font_scale: OpenCV font scale.
    """

    def __init__(
        self,
        image_dirs: Dict[str, str],
        output_dir: str = "output/edge_cases",
        line_thickness: int = 3,
        font_scale: float = 0.6,
    ) -> None:
        self.image_dirs = {k: Path(v) for k, v in image_dirs.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self._rendered_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_all(self, flagged_samples: List[FlaggedSample]) -> List[Path]:
        """Render bounding boxes for all flagged samples.

        Groups samples by image so that multiple flags on the same image
        are drawn together.

        Args:
            flagged_samples: List of flagged samples from :class:`AnomalyDetector`.

        Returns:
            List of paths to saved annotated images.
        """
        self._rendered_paths.clear()

        # Group by (file_name, split) to avoid loading the same image twice
        grouped: Dict[Tuple[str, str], List[FlaggedSample]] = {}
        for sample in flagged_samples:
            key = (sample.file_name, sample.split)
            grouped.setdefault(key, []).append(sample)

        logger.info(
            "Rendering %d flagged samples across %d images …",
            len(flagged_samples),
            len(grouped),
        )

        for (file_name, split), samples in grouped.items():
            out_path = self._render_image(file_name, split, samples)
            if out_path is not None:
                self._rendered_paths.append(out_path)

        logger.info(
            "Saved %d edge-case images to %s",
            len(self._rendered_paths),
            self.output_dir,
        )
        return self._rendered_paths

    @property
    def rendered_paths(self) -> List[Path]:
        """All rendered image paths from the last ``render_all`` call."""
        return list(self._rendered_paths)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_image_path(self, file_name: str, split: str) -> Optional[Path]:
        """Resolve the full image path given a filename and split.

        Args:
            file_name: Image filename (e.g. ``b1c66a42-6f7d68ca.jpg``).
            split: Dataset split name.

        Returns:
            Resolved Path or None if not found.
        """
        # Try the split-specific directory first
        if split in self.image_dirs:
            candidate = self.image_dirs[split] / file_name
            if candidate.exists():
                return candidate

        # Fall back: try all directories
        for dir_path in self.image_dirs.values():
            candidate = dir_path / file_name
            if candidate.exists():
                return candidate

        return None

    def _render_image(
        self,
        file_name: str,
        split: str,
        samples: List[FlaggedSample],
    ) -> Optional[Path]:
        """Load an image, draw boxes for all flagged samples, and save.

        Args:
            file_name: Image filename.
            split: Dataset split.
            samples: Flagged samples to draw on this image.

        Returns:
            Path to saved annotated image, or None on failure.
        """
        img_path = self._resolve_image_path(file_name, split)
        if img_path is None:
            logger.warning("Image not found: %s (split=%s)", file_name, split)
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to read image: %s", img_path)
            return None

        for sample in samples:
            self._draw_box(img, sample)

        # Save with a descriptive filename
        # Use first sample's reason keyword for the filename prefix
        reason_tag = samples[0].reason.split()[0].lower()
        cls_tag = samples[0].category_name.replace(" ", "_")
        out_name = f"{cls_tag}_{reason_tag}_{file_name}"
        out_path = self.output_dir / out_name
        cv2.imwrite(str(out_path), img)
        return out_path

    def _draw_box(self, img: np.ndarray, sample: FlaggedSample) -> None:
        """Draw a single bounding box with label and reason on the image.

        Args:
            img: OpenCV image (BGR, will be modified in-place).
            sample: The flagged sample to draw.
        """
        x, y, w, h = sample.bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        color = _color_for_reason(sample.reason)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.line_thickness)

        # Build label text
        label = f"{sample.category_name} | {sample.reason}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, self.font_scale, 1)

        # Background rectangle for text
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(
            img,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )

        # Text (black for contrast)
        cv2.putText(
            img,
            label,
            (x1 + 2, label_y - 2),
            font,
            self.font_scale,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
