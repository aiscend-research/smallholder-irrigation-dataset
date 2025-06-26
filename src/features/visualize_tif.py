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
    return rescale_intensity(band, in_range='image', out_range=(0, 1))


def plot_rgb(image, bands=(3, 2, 1), figsize=(8, 8), title=None):
    """
    Plot an RGB composite.
    Args:
        image (np.ndarray): Array of shape (bands, height, width)
        bands (tuple): Band indices for RGB (1-based, e.g., (4, 3, 2) for Sentinel-2 true color)
    """
    # Convert to 0-based indexing
    r, g, b = (image[b - 1] for b in bands)

    rgb = np.stack([normalize_band(r), normalize_band(g), normalize_band(b)], axis=-1)

    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Example usage
    image_path = "data/features/test_s2_export.tif"
    image, profile = load_tif(image_path)

    plot_rgb(image, bands=(4, 3, 2), title="True Color Composite")