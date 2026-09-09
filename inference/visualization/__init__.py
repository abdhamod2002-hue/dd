"""Visualization helpers that consume the SAME production inference results.

This package does not run a second detector. It converts the real
Track / LitteringEventDetector / TrackStore objects into a structured
FrameAnalysis contract and renders it onto frames.
"""

from inference.visualization.frame_analysis import (
    AssociationAnalysis,
    FrameAnalysis,
    ObjectAnalysis,
    PersonAnalysis,
    build_frame_analysis,
)
from inference.visualization.tracking_visualizer import render_analysis_frame

__all__ = [
    "AssociationAnalysis",
    "FrameAnalysis",
    "ObjectAnalysis",
    "PersonAnalysis",
    "build_frame_analysis",
    "render_analysis_frame",
]
