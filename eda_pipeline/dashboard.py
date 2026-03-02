"""
dashboard.py — Self-Contained HTML Dashboard Builder.

Compiles all distribution plots, per-class analysis panels, split comparison
visuals, and rendered edge-case bounding-box images into a single,
dark-themed, responsive HTML dashboard.

Images are base64-embedded so the HTML is fully self-contained.
Each section includes engineer-facing descriptions explaining the methods
used and how to interpret the visualizations.
"""

import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _encode_image_base64(path: Path) -> str:
    """Read an image file and return its base64-encoded data-URI string.

    Args:
        path: Local path to an image file (PNG / JPG).

    Returns:
        A ``data:image/…;base64,…`` string suitable for ``<img src=…>``.
    """
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix, "image/png"
    )
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Plot descriptions (engineer-facing)
# ---------------------------------------------------------------------------
PLOT_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "class_distribution": {
        "title": "Class Distribution — Train vs Val",
        "method": "Frequency count of annotations grouped by object class and split.",
        "inference": (
            "Check for class imbalance — if minority classes have orders-of-magnitude "
            "fewer samples, consider oversampling, focal loss, or class-weighted training. "
            "Verify train/val ratio is consistent across classes to avoid evaluation bias."
        ),
    },
    "detections_per_image_cdf": {
        "title": "CDF — Detections per Image",
        "method": (
            "Empirical Cumulative Distribution Function of detected object counts per image. "
            "For each image, count annotations; sort and plot the cumulative fraction."
        ),
        "inference": (
            "If train and val curves diverge, the splits have different scene complexity. "
            "A long tail means some images are very crowded — consider NMS tuning. "
            "The median indicates typical scene density."
        ),
    },
    "area_kde": {
        "title": "Box Area KDE — Train vs Val",
        "method": (
            "Kernel Density Estimation (Gaussian kernel, Scott bandwidth) "
            "on log₁₀-transformed bounding-box areas."
        ),
        "inference": (
            "If train/val peaks misalign, there is a size distribution shift that could "
            "bias anchor generation. Multi-modal peaks ⇒ distinct size clusters. "
            "Extremely small areas (< 10² px²) may be labeling noise."
        ),
    },
    "aspect_ratio_kde": {
        "title": "Aspect Ratio KDE — Train vs Val",
        "method": "KDE on aspect ratios (width ÷ height), clipped at 10 to remove extreme outliers.",
        "inference": (
            "AR ≈ 1 means roughly square boxes. Class-specific AR peaks inform anchor-box "
            "aspect-ratio choices for detectors like Faster R-CNN, SSD, or YOLO."
        ),
    },
    "per_class_area_violin": {
        "title": "Per-Class Area Violin — Train vs Val",
        "method": (
            "Violin plot on log₁₀-transformed area, split by train (blue) and val (pink). "
            "Internal lines show quartiles (Q1, median, Q3)."
        ),
        "inference": (
            "Long left tail ⇒ many very small instances (hard to detect). "
            "If train/val shapes differ, expect domain shift in evaluation. "
            "Wide violins ⇒ high variance in object size."
        ),
    },
    "co_occurrence_heatmap": {
        "title": "Class Co-Occurrence Heatmap",
        "method": (
            "Count of how often each pair of classes co-occurs in the same image. "
            "Diagonal ⇒ total images containing that class."
        ),
        "inference": (
            "High off-diagonal values indicate contextual relationships (e.g., rider ↔ motor). "
            "Useful for multi-label heads, contextual reasoning modules, "
            "or validating that rare-class samples include realistic context."
        ),
    },
    "global_spatial_heatmap": {
        "title": "Global Spatial Heatmap — All Object Centers",
        "method": "2D histogram (64×36 bins) of bounding-box center coordinates across all classes.",
        "inference": (
            "High-density regions show where objects are most commonly annotated. "
            "Dead zones suggest the detector sees few training examples there — "
            "consider augmentation (random crop, mosaic)."
        ),
    },
}

PER_CLASS_DESCRIPTION = (
    "Each per-class analysis panel contains 6 sub-plots:\n"
    "① Area Distribution (KDE): train vs val area densities on a log₁₀ scale. "
    "Multi-modal peaks indicate distinct size clusters.\n"
    "② Aspect Ratio (KDE): w/h distribution — informs anchor box design.\n"
    "③ Spatial Heatmap: 2D histogram of box centers — shows WHERE this class appears "
    "in the image (e.g., 'car' near horizon, 'traffic light' upper-center).\n"
    "④ Position vs Size Scatter: center position colored by area — reveals "
    "perspective effects (smaller objects higher in frame).\n"
    "⑤ Width vs Height Scatter: aspect-ratio cluster shape and dimensional range.\n"
    "⑥ Instance Count per Image: crowding distribution — useful for NMS tuning."
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
:root {
    --bg: #1e1e2e;
    --surface: #24243a;
    --card: #2a2a42;
    --text: #cdd6f4;
    --text-dim: #a6adc8;
    --accent: #89b4fa;
    --pink: #f38ba8;
    --green: #a6e3a1;
    --border: #45475a;
    --radius: 12px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
}

header {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a42 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.5rem 3rem;
    text-align: center;
}
header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
header p { color: var(--text-dim); margin-top: 0.5rem; font-size: 1rem; }

.container { max-width: 1500px; margin: 0 auto; padding: 2rem; }

/* Navigation */
nav.toc {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    margin-bottom: 2rem;
}
nav.toc h3 { color: var(--accent); margin-bottom: 0.6rem; font-size: 1rem; }
nav.toc a {
    color: var(--text-dim);
    text-decoration: none;
    margin-right: 1.5rem;
    font-size: 0.9rem;
}
nav.toc a:hover { color: var(--accent); text-decoration: underline; }

/* Summary cards */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin-bottom: 3rem;
}
.summary-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
}
.summary-card .number {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
}
.summary-card .label {
    color: var(--text-dim);
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* Sections */
section { margin-bottom: 3rem; }
section h2 {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
}

/* Plot grid */
.plot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
    gap: 1.5rem;
}
.plot-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.plot-card img {
    width: 100%;
    display: block;
}
.plot-desc {
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text-dim);
    border-top: 1px solid var(--border);
}
.plot-desc .plot-title { color: var(--text); font-weight: 600; margin-bottom: 0.3rem; }
.plot-desc .plot-method { margin-bottom: 0.3rem; }
.plot-desc .plot-inference { color: var(--green); }
.plot-desc strong { color: var(--accent); }

/* Per-class analysis */
.per-class-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
}
.per-class-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.per-class-card img {
    width: 100%;
    display: block;
}

/* Edge-case gallery */
.edge-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 1.2rem;
}
.edge-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
}
.edge-card img {
    width: 100%;
    display: block;
}
.edge-card .meta {
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: var(--text-dim);
}

/* Class accordion */
.class-section { margin-bottom: 1.5rem; }
.class-section summary {
    cursor: pointer;
    font-size: 1.15rem;
    font-weight: 600;
    padding: 0.8rem 1.2rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    list-style: none;
    user-select: none;
}
.class-section summary::before { content: '▶ '; color: var(--accent); }
.class-section[open] summary::before { content: '▼ '; }
.class-section .class-content { padding: 1rem; }

/* Explanation box */
.explanation-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    color: var(--text-dim);
    line-height: 1.7;
    white-space: pre-line;
}

/* Color legend */
.color-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}
.color-legend .item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
}
.color-legend .swatch {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    display: inline-block;
}

footer {
    text-align: center;
    padding: 2rem;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
}
"""


class DashboardBuilder:
    """Generates a self-contained HTML dashboard from plots and images.

    The dashboard includes:
      - Dataset overview with summary cards
      - Split analysis with distribution plots (each with method + inference notes)
      - Per-class dedicated analysis panels (6-sub-plot pages per class)
      - Per-class anomaly edge-case gallery with bounding-box renders

    Args:
        output_dir: Root output directory (should contain ``plots/`` and
            ``edge_cases/`` sub-directories).
    """

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.per_class_dir = self.plots_dir / "per_class"
        self.edge_dir = self.output_dir / "edge_cases"

    def build(
        self,
        summary: Dict,
        edge_case_paths: Optional[List[Path]] = None,
    ) -> Path:
        """Generate the HTML dashboard.

        Args:
            summary: Dataset summary dict from ``COCODatasetParser.summary()``.
            edge_case_paths: Paths to rendered edge-case images.

        Returns:
            Path to the generated ``dashboard.html``.
        """
        logger.info("Building HTML dashboard …")

        html_parts: List[str] = []
        html_parts.append(self._html_head())
        html_parts.append(self._header_section())
        html_parts.append('<div class="container">')

        # --- Table of contents ---
        html_parts.append(self._toc())

        # --- Summary cards ---
        html_parts.append(self._summary_cards(summary))

        # --- Distribution / split analysis plots ---
        html_parts.append(self._plots_section())

        # --- Per-class analysis ---
        html_parts.append(self._per_class_section())

        # --- Edge cases ---
        if edge_case_paths:
            html_parts.append(self._edge_cases_section(edge_case_paths))

        html_parts.append("</div>")  # .container
        html_parts.append(self._footer())
        html_parts.append("</body></html>")

        out_path = self.output_dir / "dashboard.html"
        out_path.write_text("\n".join(html_parts), encoding="utf-8")
        logger.info("Dashboard saved to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # HTML building blocks
    # ------------------------------------------------------------------

    def _html_head(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Object Detection EDA Dashboard</title>
<style>{_CSS}</style>
</head>
<body>"""

    def _header_section(self) -> str:
        return """<header>
<h1>🔍 Object Detection — Exploratory Data Analysis</h1>
<p>Automated distribution analysis, split comparison, per-class profiling, and anomaly detection</p>
</header>"""

    def _toc(self) -> str:
        return """<nav class="toc">
<h3>📋 Sections</h3>
<a href="#overview">Dataset Overview</a>
<a href="#split-analysis">Split Analysis &amp; Distributions</a>
<a href="#per-class">Per-Class Detailed Analysis</a>
<a href="#anomalies">Per-Class Anomalies &amp; Edge Cases</a>
</nav>"""

    def _summary_cards(self, summary: Dict) -> str:
        cards = []
        cards.append(
            self._card(f"{summary.get('total_images', 0):,}", "Total Images")
        )
        cards.append(
            self._card(
                f"{summary.get('total_annotations', 0):,}", "Total Annotations"
            )
        )
        cards.append(
            self._card(str(len(summary.get("classes", []))), "Object Classes")
        )
        for split_name, split_info in summary.get("splits", {}).items():
            cards.append(
                self._card(
                    f"{split_info.get('images', 0):,}",
                    f"{split_name.capitalize()} Images",
                )
            )
            cards.append(
                self._card(
                    f"{split_info.get('annotations', 0):,}",
                    f"{split_name.capitalize()} Annotations",
                )
            )
        return f"""<section id="overview">
<h2>Dataset Overview</h2>
<div class="summary-grid">{''.join(cards)}</div>
</section>"""

    @staticmethod
    def _card(number: str, label: str) -> str:
        return f"""<div class="summary-card">
<div class="number">{number}</div>
<div class="label">{label}</div>
</div>"""

    def _plots_section(self) -> str:
        """Embed all PNGs from the plots directory with descriptions."""
        if not self.plots_dir.exists():
            return ""

        # Only include top-level plots (not per_class subfolder)
        plot_files = sorted(
            p for p in self.plots_dir.glob("*.png") if not p.name.startswith("._")
        )
        if not plot_files:
            return ""

        cards = []
        for pf in plot_files:
            b64 = _encode_image_base64(pf)
            stem = pf.stem
            desc = PLOT_DESCRIPTIONS.get(stem)

            if desc:
                desc_html = (
                    f'<div class="plot-desc">'
                    f'<div class="plot-title">{desc["title"]}</div>'
                    f'<div class="plot-method"><strong>Method:</strong> {desc["method"]}</div>'
                    f'<div class="plot-inference"><strong>How to read:</strong> {desc["inference"]}</div>'
                    f"</div>"
                )
            else:
                desc_html = (
                    f'<div class="plot-desc">'
                    f'<div class="plot-title">{stem.replace("_", " ").title()}</div>'
                    f"</div>"
                )

            cards.append(
                f'<div class="plot-card">'
                f'<img src="{b64}" alt="{stem}" loading="lazy">'
                f"{desc_html}</div>"
            )

        return f"""<section id="split-analysis">
<h2>Split Analysis &amp; Distributions</h2>
<div class="explanation-box">
These plots compare the training and validation splits across multiple dimensions.
The goal is to check for <strong>distribution shifts</strong> between splits that could bias model evaluation,
identify <strong>class imbalance</strong>, and inform <strong>anchor box design</strong> decisions.
Blue = Train, Pink = Val.
</div>
<div class="plot-grid">{''.join(cards)}</div>
</section>"""

    def _per_class_section(self) -> str:
        """Embed per-class analysis panels from the per_class/ subdirectory."""
        if not self.per_class_dir.exists():
            return ""

        class_files = sorted(
            p
            for p in self.per_class_dir.glob("*_analysis.png")
            if not p.name.startswith("._")
        )
        if not class_files:
            return ""

        panels = []
        for cf in class_files:
            b64 = _encode_image_base64(cf)
            cls_name = cf.stem.replace("_analysis", "").replace("_", " ").title()
            panels.append(
                f'<div class="per-class-card">'
                f'<img src="{b64}" alt="{cls_name} analysis" loading="lazy">'
                f"</div>"
            )

        return f"""<section id="per-class">
<h2>Per-Class Detailed Analysis</h2>
<div class="explanation-box">{PER_CLASS_DESCRIPTION}</div>
<div class="per-class-grid">{''.join(panels)}</div>
</section>"""

    def _edge_cases_section(self, paths: List[Path]) -> str:
        """Embed edge-case images grouped by class with color legend."""
        # Color legend
        legend_items = [
            ("LARGEST", "#64c864", "Largest bounding boxes for the class"),
            ("SMALLEST", "#ffb400", "Smallest bounding boxes for the class"),
            ("TALLEST", "#6464ff", "Most extreme tall/skinny aspect ratios"),
            ("WIDEST", "#ff6464", "Most extreme wide/flat aspect ratios"),
            ("POSITION", "#ff00ff", "Object at unusual image location for its class"),
            ("OUTLIER", "#ff0000", "Statistical outlier (IQR + Isolation Forest)"),
            ("CROWDED", "#00c8ff", "From the most crowded image for the class"),
        ]
        legend_html = '<div class="color-legend">'
        for label, color, tooltip in legend_items:
            legend_html += (
                f'<div class="item" title="{tooltip}">'
                f'<span class="swatch" style="background:{color}"></span>'
                f"{label}</div>"
            )
        legend_html += "</div>"

        # Group by class
        grouped: Dict[str, List[Path]] = {}
        for p in paths:
            cls_name = p.stem.split("_")[0]
            grouped.setdefault(cls_name, []).append(p)

        sections = []
        for cls_name in sorted(grouped.keys()):
            images_html = []
            for img_path in grouped[cls_name]:
                b64 = _encode_image_base64(img_path)
                meta_text = img_path.stem.replace("_", " ")
                images_html.append(
                    f'<div class="edge-card">'
                    f'<img src="{b64}" alt="{img_path.stem}" loading="lazy">'
                    f'<div class="meta">{meta_text}</div></div>'
                )

            sections.append(
                f'<details class="class-section" open>'
                f"<summary>{cls_name.replace('_', ' ').title()} "
                f"({len(grouped[cls_name])} images)</summary>"
                f'<div class="class-content">'
                f'<div class="edge-gallery">{"".join(images_html)}</div>'
                f"</div></details>"
            )

        return f"""<section id="anomalies">
<h2>Per-Class Anomalies &amp; Edge Cases</h2>
<div class="explanation-box">
Anomaly detection is performed <strong>independently for each class</strong> using:
• <strong>IQR (Interquartile Range):</strong> Flags objects whose area or aspect ratio falls outside 1.5× IQR fences.
• <strong>Isolation Forest:</strong> ML-based anomaly detection on 6 features (area, aspect_ratio, bbox_w, bbox_h, center_x, center_y).
• <strong>Position outliers:</strong> Objects appearing at unusual image locations (5th–95th percentile IQR on center coordinates).
• <strong>Extremes:</strong> Top-3 largest, smallest, tallest (skinniest), and widest (flattest) objects per class.
• <strong>Crowded:</strong> The single image containing the most instances of each class.

Each bounding box is rendered on the original image with a color-coded outline and label.
</div>
{legend_html}
{''.join(sections)}
</section>"""

    @staticmethod
    def _footer() -> str:
        return """<footer>
<p>Generated by the Object Detection EDA Pipeline — Automated Analysis, Per-Class Profiling &amp; Anomaly Detection</p>
</footer>"""
