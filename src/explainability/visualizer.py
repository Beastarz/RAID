"""Small model-independent facade for explainability rendering."""

from .rendering import (
    RenderedImage,
    colorize_heatmap,
    overlay_heatmap,
    render_residual_magnitude,
    render_signed_residual,
    side_by_side_panel,
)

__all__ = [
    "RenderedImage",
    "colorize_heatmap",
    "overlay_heatmap",
    "render_residual_magnitude",
    "render_signed_residual",
    "side_by_side_panel",
]
