# Current Contents
1. A notebook to prototype running a tree-based model using dataset/datamodule
 * I was having a hard time connecting to the SSH remote, so I decided to download the terratorch multi-temporal-crop-dataset on my local machine. I will include these files in the .gitignore for now(Around 1GB). Using tutorial from terratorch, I downloaded the a sample of the dataset from a google drive folder. 
 * Converting the data from time-series format to tabular 
    * The current format of the data in the dataset is 
    ```
    sample = {
    'image': Tensor of shape (C, T, H, W),
    'mask' : Tensor of shape (H, W)
    }
    ```
        * Where the image is the pixel information and mask is the label (i.e. crop type)
        * C is the spectral bands (color)
        * T is time steps 
        * H is height, W is width 
    * We want each pixel to be a row essentially, so we flatten the tensors by multiplying (H * W, T * C). Going from 4D to 2D
    * Since we flattened, each pixel is now a row, but we lost spatial information(location of the pixel)
 * Ran randomForest on 1000 pixels, since it was taking too long when I tried running it on everything and made predictions.
    * Can maybe downsample the pixel dimensions if needed




`build_features.py`
 """
    Flattens a TerraTorch-style dataset into a tabular format suitable for classical ML models.

    Parameters:
    ----------
    dataset : torch.utils.data.Dataset
        The dataset where each sample is a dict with keys 'image' and 'mask'.
    ignore_index : int, optional
        The label value to ignore (e.g., -1), by default -1.

    Returns:
    -------
    X : np.ndarray
        The flattened feature matrix of shape (num_pixels_total, T*C).
    y : np.ndarray
        The corresponding label vector of shape (num_pixels_total,).
"""    