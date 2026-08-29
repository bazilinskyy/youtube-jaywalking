"""Visualization and plotting utilities compliant with project export standards.

Provides standardized Plotly chart creation with automated multi-format export
to interactive HTML, raster PNG, and publication vector formats (PDF, SVG).
Note: EPS export is deprecated/unsupported in modern Kaleido (>=0.2.0); PDF and SVG
are used as standard vector alternatives.
"""

import logging
import os
from typing import Dict, List, Optional
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def save_figure_multiformat(
    fig: go.Figure,
    base_output_path: str,
    width: int = 1000,
    height: int = 600,
    scale: float = 2.0,
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Saves a Plotly figure to multiple standardized formats (HTML, PNG, PDF, SVG).

    Args:
        fig: The Plotly Figure object to export.
        base_output_path: File path without extension (e.g. 'results/visualizations/plot').
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Scale factor for high-resolution static rendering.
        formats: List of formats to export. Defaults to ['html', 'png', 'pdf', 'svg'].

    Returns:
        Dictionary mapping format names to their generated output filepaths.

    Raises:
        ValueError: If an unsupported format is requested or figure is invalid.
    """
    if formats is None:
        formats = ["html", "png", "pdf", "svg"]

    parent_dir = os.path.dirname(os.path.abspath(base_output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    saved_paths: Dict[str, str] = {}

    for fmt in formats:
        fmt_lower = fmt.lower().strip()

        if fmt_lower == "html":
            html_path = f"{base_output_path}.html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            saved_paths["html"] = html_path

        elif fmt_lower == "png":
            png_path = f"{base_output_path}.png"
            fig.write_image(png_path, width=width, height=height, scale=scale)
            saved_paths["png"] = png_path

        elif fmt_lower == "pdf":
            pdf_path = f"{base_output_path}.pdf"
            fig.write_image(pdf_path, width=width, height=height)
            saved_paths["pdf"] = pdf_path

        elif fmt_lower == "svg":
            svg_path = f"{base_output_path}.svg"
            fig.write_image(svg_path, width=width, height=height)
            saved_paths["svg"] = svg_path

        elif fmt_lower == "eps":
            logger.warning(
                "EPS export requested for %s, but EPS is deprecated/unsupported by Kaleido. "
                "Falling back to vector PDF and SVG export.",
                base_output_path,
            )
            # Generate PDF and SVG as vector substitutes for EPS
            pdf_fallback = f"{base_output_path}.pdf"
            fig.write_image(pdf_fallback, width=width, height=height)
            saved_paths["pdf"] = pdf_fallback

            svg_fallback = f"{base_output_path}.svg"
            fig.write_image(svg_fallback, width=width, height=height)
            saved_paths["svg"] = svg_fallback

        else:
            raise ValueError(
                f"Unsupported figure export format '{fmt}'. "
                f"Supported formats are: 'html', 'png', 'pdf', 'svg'."
            )

    return saved_paths
