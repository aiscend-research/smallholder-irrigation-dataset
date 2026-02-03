"""
DEPRECATED: This module is a backward-compatibility wrapper.
Use satellite_visualization.py instead, which supports both Sentinel-2 and PlanetScope.

All functions are re-exported from satellite_visualization for backward compatibility.
"""

# Re-export all functions for backward compatibility
from .satellite_visualization import (
    # Configuration
    SENSOR_CONFIG,
    IRRIGATION_COLORS,
    IRRIGATION_LABELS,
    LABELER_COLORS,
    LABELER_COLORS_HEX,

    # Path functions
    get_features_dir,
    get_irrigation_table_path,

    # Stack/label finding
    find_stack_for_site,
    find_labels_for_stack,
    find_matching_stack_for_screenshot,
    get_labeled_timestep,

    # Loading functions
    load_rgb_from_stack,
    load_label_mask,

    # Visualization functions
    plot_satellite_with_mask,

    # Utility functions
    trace_pixel_boundaries,
)

# Legacy constants for backward compatibility
BAND_INDICES = SENSOR_CONFIG['sentinel2']['band_indices']
