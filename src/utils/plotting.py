"""Visualization and plotting utilities compliant with project export standards.

Provides standardized Plotly chart creation with automated multi-format export
to HTML, PNG, and EPS/PDF vector formats.
"""

import os
from typing import Dict, List, Optional
import plotly.graph_objects as go


def save_figure_multiformat(
    fig: go.Figure,
    base_output_path: str,
    width: int = 1000,
    height: int = 600,
    scale: float = 2.0,
    formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Saves a Plotly figure to multiple standardized formats (HTML, PNG, EPS/PDF).

    Args:
        fig: The Plotly Figure object to export.
        base_output_path: File path without extension (e.g. 'results/visualizations/accuracy_plot').
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Scale factor for high-resolution static rendering.
        formats: List of formats to export. Defaults to ['html', 'png', 'eps'].

    Returns:
        Dictionary mapping format names to their generated output filepaths.
    """
    if formats is None:
        formats = ["html", "png", "eps"]

    os.makedirs(os.path.dirname(os.path.abspath(base_output_path)), exist_ok=True)
    saved_paths: Dict[str, str] = {}

    # 1. Interactive HTML Export
    if "html" in formats:
        html_path = f"{base_output_path}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        saved_paths["html"] = html_path

    # 2. Static PNG Export (via kaleido)
    if "png" in formats:
        png_path = f"{base_output_path}.png"
        try:
            fig.write_image(png_path, width=width, height=height, scale=scale)
            saved_paths["png"] = png_path
        except Exception:
            pass

    # 3. Vector EPS / PDF Export
    if "eps" in formats:
        eps_path = f"{base_output_path}.eps"
        try:
            fig.write_image(eps_path, width=width, height=height)
            saved_paths["eps"] = eps_path
        except Exception:
            # Fallback to PDF if EPS encoder is unavailable
            pdf_path = f"{base_output_path}.pdf"
            try:
                fig.write_image(pdf_path, width=width, height=height)
                saved_paths["pdf"] = pdf_path
            except Exception:
                pass

    return saved_paths
