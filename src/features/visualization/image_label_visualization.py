from glob import glob
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.plot import show
from skimage import measure
from matplotlib.path import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from ...utils.utils import get_data_root, find_project_root
import pandas as pd
import re
import sys


# Refer to src/features/README.md for dataset version info
FEATURES_VERSION = "features_v2"

# Directories to search for labels & satellite images
LABEL_DIR = get_data_root() + f"dataset/labels/"
SAT_DIR = get_data_root() + f"{FEATURES_VERSION}/_tmp_tif/"
LABEL_CSV = "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"

'''
Helper function that converts bands from range [0,10000] to [0, 1] for 
visualization, through a percentile stretch. This is necessary because
Google Earth Engine stores values as integers range [0, 10000] to 
preserve precision

Parameters:
    - channel_int: Raw band values.

Return:
    - float32 in [0, 1]; NO_DATA left as 1 (white) for now.
'''
def _stretch_01(channel_int, p_lo=2, p_hi=98):

    ch = channel_int.astype(np.float32) / 10000.0
    mask = (channel_int == -9999)
    vals = ch[~mask]
    if vals.size == 0:
        out = np.zeros_like(ch, dtype=np.float32)
        out[mask] = np.nan
        return out

    lo, hi = np.percentile(vals, (p_lo, p_hi))
    if hi <= lo + 1e-6:
        # Avoid divide-by-zero
        out = np.clip(ch, 0.0, 1.0)
    else:
        out = (ch - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)
    out[mask] = np.nan
    return out

'''
Visualizes the labels on top of the RGB satellite image. Utilizes skimage.measure
to find the outlines of each label polygon, and plots them on top of the RGB satellite 
image. Creates and saves one .png file under "data/visualizations/{img_id}_label_visualization.png"
Each .png file contains 8 plots, one for each label band.

Params:
    - labels (np.array): A numpy array of labels
    - rgb (np.array): A numpy array of RGB values for the satellite image.
    - img_id (int): The unique ID of this label/image pair.
'''
def visualize(labels, rgb, ndvi, img_id):
    # Define discrete colormap for class values (e.g., 0–5)
    colors = ['black', 'blue', 'green', 'yellow', 'orange', 'red']
    cmap = ListedColormap(colors)

    # Create a 4x2 grid for plotting 8 bands
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))  # Width x Height in inches
    axes = axes.flatten()

    band_dict = {
        1: "Categorical Irrigation Type Classification",
        2: "Binary Irrigation Classification",
        3: "Explanation: Unclear signs of agriculture",
        4: "Explanation: Only slightly green",
        5: "Explanation: Uneven",
        6: "Explanation: May naturally be green",
        7: "Explanation: May be a fishpond",
        8: "Certainty score"
    }

    # Plot each band
    fig.suptitle(f"Satellite Image with Label Overlays (id {img_id})", fontsize=18, x=0.6, va='center')

    # Plots labels on RGB
    for i in range(8):
        ax = axes[i]
        ax.imshow(rgb)  # draw satellite background

        label_band = labels[i]
        masked_label = np.ma.masked_where(label_band == 0, label_band)
        len_colors = len(colors) if i == 7 or i == 0 else 2

        cmap = ListedColormap(colors[:len_colors])
        im = ax.imshow(masked_label, cmap=cmap, vmin=0, vmax=len_colors - 1, alpha=0)

        # --- Draw outlines around labeled regions, colored by dominant pixel value ---
        contours = measure.find_contours(label_band, level=0.5)

        for contour in contours:
            path = Path(contour)

            # Get all pixel coordinates
            y, x = np.mgrid[0:label_band.shape[0], 0:label_band.shape[1]]
            coords = np.vstack((y.flatten(), x.flatten())).T

            # Find pixels inside the contour
            inside = path.contains_points(coords).reshape(label_band.shape)
            vals_inside = label_band[inside]
            vals_inside = vals_inside[vals_inside > 0]
            if len(vals_inside) == 0:
                continue

            # Find the most common class (dominant value)
            cls = np.bincount(vals_inside.astype(int)).argmax()

            # Get corresponding color from cmap
            color = cmap(cls / (len_colors - 1))[:3]

            # Draw the outline with that color
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5)

        mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=len_colors - 1), cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, ticks=range(len_colors))

        ax.set_title(f'Label Band {i+1} ({band_dict[i+1]})')
        ax.axis('off')

        plt.tight_layout()

        # Create visualizations directory if it doesn't already exist
        png_path = f"data/visualizations/{img_id}_label_visualization.png"
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path)

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))  # Width x Height in inches
    axes = axes.flatten()

    # Plot each band
    fig.suptitle(f"Satellite NDVI with Label Overlays (id {img_id})", fontsize=18, x=0.6, va='center')

    # Plots labels on NDVI
    for i in range(8):
        ax = axes[i]
        ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)  # draw NDVI background

        label_band = labels[i]
        masked_label = np.ma.masked_where(label_band == 0, label_band)
        len_colors = len(colors) if i == 7 or i == 0 else 2

        cmap = ListedColormap(colors[:len_colors])
        im = ax.imshow(masked_label, cmap=cmap, vmin=0, vmax=len_colors - 1, alpha=0)

        # --- Draw outlines around labeled regions, colored by dominant pixel value ---
        contours = measure.find_contours(label_band, level=0.5)

        for contour in contours:
            path = Path(contour)

            # Get all pixel coordinates
            y, x = np.mgrid[0:label_band.shape[0], 0:label_band.shape[1]]
            coords = np.vstack((y.flatten(), x.flatten())).T

            # Find pixels inside the contour
            inside = path.contains_points(coords).reshape(label_band.shape)
            vals_inside = label_band[inside]
            vals_inside = vals_inside[vals_inside > 0]
            if len(vals_inside) == 0:
                continue

            # Find the most common class (dominant value)
            cls = np.bincount(vals_inside.astype(int)).argmax()

            # Get corresponding color from cmap
            color = cmap(cls / (len_colors - 1))[:3]

            # Draw the outline with that color
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5)

        mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=len_colors - 1), cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, ticks=range(len_colors))

        ax.set_title(f'Label Band {i+1} ({band_dict[i+1]})')
        ax.axis('off')

        plt.tight_layout()

        # Create visualizations directory if it doesn't already exist
        png_path = f"data/visualizations/{img_id}_label_visualization_ndvi.png"
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path)

'''
Given the path of the label and satellite TIF files, reads in the files as np.arrays
and prepares them for visualization.

Params:
    - label_path (str):
    - sat_image_path (str):

Returns:
    - image (np.array): The label TIF image as a numpy array.
    - rgb (np.array): The RGB values of the satellite image as an array
    - ndvi (np.array): The NDVI values of the satellite image as an array
'''
def retrieve_images(label_path, sat_image_path, uid):
    sat_img = None
    with rasterio.open(sat_image_path) as src:
        sat_img = src.read(list(range(1, 11)))  # Read RGB & NDVI bands

    r = _stretch_01(sat_img[2])
    g = _stretch_01(sat_img[0])
    b = _stretch_01(sat_img[1])
    ndvi = retrieve_ndvi(sat_img)

    rgb = np.stack([r, g, b], axis=-1)  # (H,W,3)
    rgb = np.where(np.isnan(rgb), 1, rgb)  # Replace NaNs with 1 (white) for visualization

    # Load TIF file
    with tifffile.TiffFile(label_path) as tif:
        image = tif.asarray()

    # Ensure shape is (8, 100, 100)
    if image.shape[-1] == 8:
        image = np.transpose(image, (2, 0, 1))

    return image, rgb, ndvi

def retrieve_ndvi(sat_img):
    # NEED TO CALCULATE NDVI -> NOT SAVED IN THE RAW IMAGES...
    # RAW IMAGE BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12', 'SCL']
    def m(a_int):
        return np.ma.masked_equal(a_int, -9999).astype(np.float32) / 10000.0
    
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [m(b) for b in sat_img[:10]]
    ndvi = (B8 - B4) / (B8 + B4)
    def to_int16(ma):
        ma = np.ma.clip(ma, -1.0, 1.0)
        out = np.full(ma.shape, -9999, dtype=np.int16)
        valid = ~np.ma.getmaskarray(ma)
        out[valid] = (ma[valid] * 10000.0).astype(np.int16)
        return out

    ndvi = to_int16(ndvi)
    ndvi = ndvi.astype(np.float32) / 10000.0
    ndvi[ndvi == -9999] = np.nan
    return ndvi
    
'''
Given a unique ID of an labelled image, searches for and returns the label TIF file and 
corresponding satellite image in the data directory. Throws FileNotFoundError if no 
corresponding file found, RuntimeError if multiple files found.

Params:
    - uid (int): The unique image ID of an image

Returns:
    - label_path (str): The path of the label file for uid.
    - sat_path (str): The path of the satellite image for uid.

'''
def find_files_for_id(uid):
    # Search for label file
    matching_files = glob(os.path.join(LABEL_DIR, f"{uid}_*label.tif"))
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No matching .tif file found for ID {uid}")

    if len(matching_files) > 1:
        raise RuntimeError(f"Multiple matching files found for ID {uid}:", str(matching_files))
    

    label_path = matching_files[0]

    # Determine which of the 37 images has a matching date
    # Format of satellite image file name: s2_{y}_{x}_yyyy-mm-dd_yyyy-mm-dd.tif 
    matching_files = glob(os.path.join(SAT_DIR, f"site_*_{uid}/s2*.tif"))
    
    # Determine date of image through regex matching on file name.
    # Format of label file name: {uid}_{site_id}_yyyy.mm.dd_{labeler_initials}_label.tif

    filename = label_path.split("/")[-1]
    match = re.search(r"\d{4}\.\d{2}\.\d{2}", filename)
    image_date = None

    if match:
        image_date = match.group(0)

    else:
        raise RuntimeError(f"Unable to find image date for ID {uid}")
    
    # Determine matching satellite image
    sat_img_path = None
    masked_files = [f for f in matching_files if "masked" in f]
    dates = [re.findall(r"\d{4}-\d{2}-\d{2}", f) for f in masked_files]

    for idx, d in enumerate(dates):
        try: 
            a, b = d
            a = a.replace("-", ".")
            b = b.replace("-", ".")

        except:
            raise RuntimeError(f"Unable to parse start and end dates from satellite image {masked_files[idx]}")
        
        # Found match!
        if a <= image_date and b >= image_date:
            sat_img_path = masked_files[idx]
            break

    if not sat_img_path:
        raise RuntimeError(f"Unable to find corresponding satellite image for ID {uid}")

    return label_path, sat_img_path



'''
Select a random sample of uids to plot, with an option to select images with at least one labeled polygon.
Does this by filtering data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv for images with 
greater than 0 percent coverage, and randomly selecting remaining uids.

Params:
    - n (int, optional): Number of samples, default 10
    - irrigation (bool, optional): Option to select images with at least one labeled polygon. Default True.
    - certainty_thresh (int, optional): Optionally, select only images w/ labeled polygons of certainty 
    greater than certainty_thresh. Default 3.

Returns:
    - uids (list): A list of selected UIDs.
'''
def select_uids(n=10, irrigation=True, certainty_thresh=1):
    table = pd.read_csv(LABEL_CSV)
    if irrigation:
        table = table[table['percent_coverage'] > 0.0]

        if certainty_thresh:
            table = table[table['irrigation'] > certainty_thresh]
    
    # Randomly select n uids
    uids = table['unique_id'].sample(n=n).tolist()
    return uids

if __name__ == "__main__":

    uids = select_uids(10, True, 3)
    
    # Visualize single UID
    if len(sys.argv) > 1:
        try:
            uids = [int(sys.argv[1])]
        except ValueError:
            raise SystemExit("Usage (optionally specify UID): python3 src/features/image_label_visualization.py [uid]")

    paths = {uid : [*find_files_for_id(uid)] for uid in uids}

    for img_id, arr in paths.items():
        label_path, sat_path = arr
        label_path = label_path
        sat_path = sat_path
        labels, rgb, ndvi = retrieve_images(label_path, sat_path, img_id)
        visualize(labels, rgb, ndvi, img_id)