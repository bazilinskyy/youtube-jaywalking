"""Unit tests for Plotly multi-format figure exporting utilities."""

import os
import tempfile
import unittest
import plotly.graph_objects as go

from src.utils.plotting import save_figure_multiformat


class TestPlottingUtilities(unittest.TestCase):
    """Tests for Plotly multi-format figure export functionality."""

    def setUp(self) -> None:
        """Create a standard test figure and temporary output directory."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.fig = go.Figure(data=go.Bar(x=["A", "B", "C"], y=[10, 25, 15]))
        self.fig.update_layout(title="Test Figure")

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.test_dir.cleanup()

    def test_save_figure_default_formats(self) -> None:
        """Tests standard multi-format export across HTML, PNG, PDF, and SVG."""
        base_path = os.path.join(self.test_dir.name, "default_plot")
        saved = save_figure_multiformat(self.fig, base_path)

        # Check all standard formats are present in returned dictionary
        self.assertIn("html", saved)
        self.assertIn("png", saved)
        self.assertIn("pdf", saved)
        self.assertIn("svg", saved)

        # Check all files exist and have non-zero size
        for fmt, path in saved.items():
            self.assertTrue(os.path.isfile(path), f"File missing for format {fmt}: {path}")
            size = os.path.getsize(path)
            self.assertGreater(size, 0, f"File empty for format {fmt}: {path}")

    def test_save_figure_specific_formats(self) -> None:
        """Tests export with a restricted list of formats."""
        base_path = os.path.join(self.test_dir.name, "subset_plot")
        saved = save_figure_multiformat(self.fig, base_path, formats=["html", "png"])

        self.assertIn("html", saved)
        self.assertIn("png", saved)
        self.assertNotIn("pdf", saved)
        self.assertNotIn("svg", saved)
        self.assertTrue(os.path.isfile(saved["html"]))
        self.assertTrue(os.path.isfile(saved["png"]))

    def test_save_figure_eps_fallback(self) -> None:
        """Tests that requesting deprecated EPS gracefully falls back to vector PDF and SVG."""
        base_path = os.path.join(self.test_dir.name, "eps_plot")
        saved = save_figure_multiformat(self.fig, base_path, formats=["eps"])

        # EPS should gracefully export PDF and SVG fallbacks
        self.assertIn("pdf", saved)
        self.assertIn("svg", saved)
        self.assertTrue(os.path.isfile(saved["pdf"]))
        self.assertTrue(os.path.isfile(saved["svg"]))

    def test_save_figure_invalid_format_raises(self) -> None:
        """Tests that requesting an unknown format raises a clear ValueError."""
        base_path = os.path.join(self.test_dir.name, "invalid_plot")
        with self.assertRaises(ValueError) as context:
            save_figure_multiformat(self.fig, base_path, formats=["docx"])
        self.assertIn("Unsupported figure export format", str(context.exception))
