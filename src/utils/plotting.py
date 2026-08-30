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

# Logger instance for emission of export status and deprecation warnings
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
        width: Image width in pixels for raster and vector rendering.
        height: Image height in pixels for raster and vector rendering.
        scale: Scale factor for high-resolution static rendering (e.g., 2.0 = 2x DPI).
        formats: List of formats to export. Defaults to ['html', 'png', 'pdf', 'svg'].

    Returns:
        Dictionary mapping format names to their generated output filepaths.

    Raises:
        ValueError: If an unsupported format is requested or figure is invalid.
    """
    # Default to standard publication bundle containing web interactive, raster, and vector formats
    if formats is None:
        formats = ["html", "png", "pdf", "svg"]

    # Extract target directory path from base filename
    parent_dir = os.path.dirname(os.path.abspath(base_output_path))
    # Ensure parent directory exists before writing to prevent I/O errors
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Initialize tracking dictionary for generated filepaths
    saved_paths: Dict[str, str] = {}

    # Iterate through all requested output formats sequentially
    for fmt in formats:
        # Normalize format name to lowercase and strip whitespace
        fmt_lower = fmt.lower().strip()

        # 1. Interactive Web HTML export with bundled/CDN plotly.js
        if fmt_lower == "html":
            html_path = f"{base_output_path}.html"
            # Write stand-alone HTML file referencing CDN bundle to minimize file size
            fig.write_html(html_path, include_plotlyjs="cdn")
            saved_paths["html"] = html_path

        # 2. Raster PNG export for quick viewing, presentations, and Markdown embedding
        elif fmt_lower == "png":
            png_path = f"{base_output_path}.png"
            # Render raster image at high DPI scale factor
            fig.write_image(png_path, width=width, height=height, scale=scale)
            saved_paths["png"] = png_path

        # 3. Vector PDF export for camera-ready papers and publication manuscripts
        elif fmt_lower == "pdf":
            pdf_path = f"{base_output_path}.pdf"
            # Write high-precision vector PDF
            fig.write_image(pdf_path, width=width, height=height)
            saved_paths["pdf"] = pdf_path

        # 4. Vector SVG export for web presentations, Illustrator, and vector graphics
        elif fmt_lower == "svg":
            svg_path = f"{base_output_path}.svg"
            # Write scalable vector graphic
            fig.write_image(svg_path, width=width, height=height)
            saved_paths["svg"] = svg_path

        # 5. Handle deprecated EPS format gracefully via PDF/SVG fallbacks
        elif fmt_lower == "eps":
            # Emit informative warning regarding upstream Kaleido deprecation
            logger.warning(
                "EPS export requested for %s, but EPS is deprecated/unsupported by Kaleido. "
                "Falling back to vector PDF and SVG export.",
                base_output_path,
            )
            # Generate PDF as first vector substitute for EPS
            pdf_fallback = f"{base_output_path}.pdf"
            fig.write_image(pdf_fallback, width=width, height=height)
            saved_paths["pdf"] = pdf_fallback

            # Generate SVG as second vector substitute for EPS
            svg_fallback = f"{base_output_path}.svg"
            fig.write_image(svg_fallback, width=width, height=height)
            saved_paths["svg"] = svg_fallback

        else:
            # Raise an explicit error for unknown formats
            raise ValueError(
                f"Unsupported figure export format '{fmt}'. "
                f"Supported formats are: 'html', 'png', 'pdf', 'svg'."
            )

    # Return dictionary of successfully exported format targets
    return saved_paths
