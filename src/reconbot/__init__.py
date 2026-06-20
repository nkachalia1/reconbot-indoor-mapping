"""ReconBot indoor spatial reconstruction primitives."""

from .colmap import ColmapRunReport, ColmapSparseConfig, run_sparse_colmap_wsl
from .manifest import DatasetManifest, MetricReference, load_manifest, save_manifest
from .scale import DimensionCheck, MetricScaleReport, build_metric_scale_report
from .stitch import (
    ConstrainedStitchReport,
    TapeCalibration,
    calibrate_tape_frame,
    stitch_metric_components,
)
from .trajectory import (
    CameraPose,
    TrajectoryReport,
    analyze_trajectory,
    parse_colmap_images_text,
    write_trajectory_artifacts,
)
from .video import (
    FrameExtractionConfig,
    FrameExtractionReport,
    VideoMetadata,
    extract_frames,
    inspect_video,
)

__all__ = [
    "ColmapRunReport",
    "ColmapSparseConfig",
    "ConstrainedStitchReport",
    "CameraPose",
    "DatasetManifest",
    "DimensionCheck",
    "FrameExtractionConfig",
    "FrameExtractionReport",
    "MetricReference",
    "MetricScaleReport",
    "TrajectoryReport",
    "TapeCalibration",
    "VideoMetadata",
    "build_metric_scale_report",
    "calibrate_tape_frame",
    "analyze_trajectory",
    "extract_frames",
    "inspect_video",
    "load_manifest",
    "parse_colmap_images_text",
    "run_sparse_colmap_wsl",
    "save_manifest",
    "stitch_metric_components",
    "write_trajectory_artifacts",
]
