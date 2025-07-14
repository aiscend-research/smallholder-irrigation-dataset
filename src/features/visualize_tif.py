import sys
import os
import argparse
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

def load_tif(filepath):
    with rasterio.open(filepath) as src:
        data = src.read()  # Shape: (bands, height, width)
        profile = src.profile
    return data, profile

def normalize_band(band):
    # Prevent division by zero and all-blank bands
    if np.all(band == 0):
        return band.astype(np.float32)
    return rescale_intensity(band, in_range='image', out_range=(0, 1))

def plot_rgb(image, bands=(4, 3, 2), figsize=(8, 8), title=None, return_fig=False):
    """
    Plot an RGB composite.
    Args:
        image (np.ndarray): Array of shape (bands, height, width)
        bands (tuple): Band indices for RGB (1-based, e.g., (4, 3, 2) for Sentinel-2 true color)
        return_fig (bool): If True, return the figure object
    """
    # Check if band indices are in range
    num_bands = image.shape[0]
    if max(bands) > num_bands or min(bands) < 1:
        print(f"Selected bands {bands} exceed available bands ({num_bands}).")
        return

    # Check for all-blank image
    if np.all(image == 0):
        print("Warning: This image is blank (all zeros).")
    
    # Convert to 0-based indexing
    try:
        r, g, b = (image[b - 1] for b in bands)
    except IndexError:
        print("Error: Specified bands out of range for this image.")
        return

    rgb = np.stack([normalize_band(r), normalize_band(g), normalize_band(b)], axis=-1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.axis('off')
    if title:
        ax.set_title(title)
    if return_fig:
        return fig
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="View Sentinel-2 .tif images as RGB composites.")
    parser.add_argument("tif_path", type=str, help="Path to .tif file")
    parser.add_argument("--bands", type=int, nargs=3, default=(4, 3, 2),
                        help="Band indices for R G B (1-based, default: 4 3 2 for Sentinel-2 true color)")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    args = parser.parse_args()

    if not os.path.exists(args.tif_path):
        print(f"File not found: {args.tif_path}")
        sys.exit(1)

    image, profile = load_tif(args.tif_path)
    print(f"Loaded {args.tif_path}")
    print(f"Shape: {image.shape} (bands, height, width)")
    print(f"Profile: {profile}")

    plot_rgb(image, bands=tuple(args.bands), title=args.title)

if __name__ == "__main__":
    main()
