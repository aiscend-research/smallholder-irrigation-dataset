from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import root_mean_squared_error, classification_report, mean_squared_error, mean_absolute_error
import calendar
import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.utils import get_data_root
from utils.utils import get_data_root
import pandas as pd
import numpy as np
import os
from itertools import product

LABEL_CSV =  "/home/madhav/smallholder-irrigation-dataset/data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"

MULTI_CLASSES = ['Not irrigated','Small-scale','Tree crop','Industrial','Lawn','Covered']
BINARY_CLASSES = ['Not irrigated','Irrigated']
UNCERTAINTY_EXPLANATIONS = [
    'Unclear signs of agriculture',
    'Only slightly green',
    'Uneven',
    'May naturally be green',
    'May be a fishpond'
]

def model_metrics(y_pred, y_test):
    """
    Computes accuracy, precision, recall, and F1 score for binary classification.
    
    Args:
        y_pred: numpy array, can be 1D (n_samples,) or 2D (n_samples, 1)
        y_test: numpy array, can be 1D (n_samples,) or 2D (n_samples, 1)
    
    Returns:
        dict: Dictionary with binary classification metrics.
    """
    # Ensure 1D arrays
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    if y_test.ndim > 1:
        y_test = y_test.ravel()
    
    # Binary classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    metrics = {
        "irrigation_presence": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
    }
    
    return metrics


# --- Feature importance export utility ---
def export_feature_importances(
    clf,
    band_names=None,
    num_timesteps=None,
    out_dir="./",
    prefix="",
    num_bands=None,
):
    """
    Exports feature importances and saves three CSVs:
      - Detailed importance (band, timestep)
      - Aggregated by band
      - Aggregated by time_step

    Args:
        clf: Trained MultiOutputClassifier wrapping RandomForest/GBM, or a single estimator with feature_importances_.
        band_names: Optional list of band names (length == num_bands). If None, names will be auto-generated (B1..Bn).
        num_timesteps: Optional int; if None, will be inferred from importances length and num_bands.
        out_dir: Directory to save csvs.
        prefix: Optional filename prefix (e.g., "fold1_").
        num_bands: Optional int to explicitly set number of bands (overrides len(band_names) if provided).
    """
    import numpy as _np
    import pandas as _pd
    import os as _os

    # Ensure output directory exists
    _os.makedirs(out_dir, exist_ok=True)

    # 1) Get raw feature importances as a 1D array
    if hasattr(clf, "estimators_"):
        # MultiOutputClassifier: average per-target importances
        all_imp = _np.array([est.feature_importances_ for est in clf.estimators_])
        importances = all_imp.mean(axis=0)
    else:
        importances = _np.asarray(getattr(clf, "feature_importances_", None))
        if importances is None:
            raise ValueError("Provided model does not expose feature_importances_.")

    # 2) Resolve num_bands and band_names
    if num_bands is None:
        if band_names is not None:
            num_bands = len(band_names)
        else:
            # We'll infer later once num_timesteps is known
            pass

    # 3) Resolve num_timesteps
    if num_timesteps is None and num_bands is not None and len(importances) % num_bands == 0:
        num_timesteps = len(importances) // num_bands

    if num_bands is None and num_timesteps is not None and len(importances) % num_timesteps == 0:
        num_bands = len(importances) // num_timesteps

    # If still unknown, last resort: square-ish guess (should not happen in normal runs)
    if num_bands is None or num_timesteps is None:
        raise ValueError(
            f"Cannot infer grid: importances length={len(importances)}, num_bands={num_bands}, num_timesteps={num_timesteps}. "
            "Pass num_bands/num_timesteps (from your YAML)."
        )

    # Build band names if missing
    if band_names is None:
        band_names = [f"B{i+1}" for i in range(num_bands)]

    # 4) Align importances length to expected grid size
    expected_len = num_bands * num_timesteps
    if len(importances) < expected_len:
        # pad with zeros (some estimators may drop constant features)
        importances = _np.concatenate([importances, _np.zeros(expected_len - len(importances))])
        print(f"[WARNING] Importances shorter than expected; padded with zeros to {expected_len}.")
    elif len(importances) > expected_len:
        print(f"[WARNING] Importances longer than expected ({len(importances)}>{expected_len}); truncating.")
        importances = importances[:expected_len]

    # 5) Construct feature grid names and DataFrame
    feature_names = [f"{band_names[b]}_t{t+1}" for t in range(num_timesteps) for b in range(num_bands)]

    df = _pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    df["band"] = df["feature"].str.extract(r"^(.*?)_t")[0]
    df["time_step"] = df["feature"].str.extract(r"_t(\d+)$")[0].astype(int)

    # 6) Aggregations
    agg_band = df.groupby("band", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    agg_time = df.groupby("time_step", as_index=False)["importance"].sum().sort_values("time_step")

    # 7) Save CSVs
    detailed_csv = _os.path.join(out_dir, f"{prefix}feature_importance_detailed.csv")
    band_csv = _os.path.join(out_dir, f"{prefix}feature_importance_by_band.csv")
    time_csv = _os.path.join(out_dir, f"{prefix}feature_importance_by_time.csv")
    df.to_csv(detailed_csv, index=False)
    agg_band.to_csv(band_csv, index=False)
    agg_time.to_csv(time_csv, index=False)
    print(f"Saved feature importances to {out_dir}")

    return df, agg_band, agg_time


# --- Feature importance heatmap plot utility ---
import matplotlib.pyplot as plt
import numpy as np

def plot_band_time_importance(
    importance_df,
    band_names=None,
    num_timesteps=None,
    figsize=(16, 6),
    title="Feature Importance by Band and Time Step",
    save_path=None
):
    """
    Plots a 2D heatmap of feature importances: band (y), time step (x), color = importance.

    Args:
        importance_df: DataFrame with columns ['band', 'time_step', 'importance']
        band_names: List of band names in correct order
        num_timesteps: Number of time steps (if not inferable)
        figsize: Size of the figure
        title: Title for the plot
        save_path: Optional path to save figure
    """
    # NEW: If a path is given instead of a DataFrame, load it
    if isinstance(importance_df, str):
        import pandas as pd
        importance_df = pd.read_csv(importance_df)

    # Get unique bands and time steps in order
    bands = band_names or sorted(importance_df['band'].unique(), key=lambda x: str(x))
    timesteps = sorted(importance_df['time_step'].unique())
    if num_timesteps is not None:
        timesteps = list(range(1, num_timesteps+1))

    # Build importance matrix
    importance_matrix = np.zeros((len(bands), len(timesteps)))
    for i, b in enumerate(bands):
        for j, t in enumerate(timesteps):
            val = importance_df[(importance_df["band"] == b) & (importance_df["time_step"] == t)]["importance"]
            importance_matrix[i, j] = val.values[0] if not val.empty else 0

    plt.figure(figsize=figsize)
    im = plt.imshow(importance_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, label='Importance')
    plt.yticks(range(len(bands)), bands)
    plt.xticks(range(len(timesteps)), [f"t{t}" for t in timesteps], rotation=90)
    plt.xlabel("Time Step")
    plt.ylabel("Band")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature importance heatmap to {save_path}")
    else:
        plt.show()

def plot_band_importance(df, band_names=None, title="Feature Importance by Band", save_path=None):
    """
    Plots a bar chart of feature importances aggregated by band.

    Args:
        df: DataFrame with columns ['band', 'importance'] or at least 'band' and 'importance'
        band_names: Optional list of band names in desired order
        title: Title for the plot
        save_path: Optional path to save the plot

    Usage:
        Pass a DataFrame with 'band' and 'importance' columns.
        If band_names is provided, bars will be ordered accordingly.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)

    if 'band' not in df.columns or 'importance' not in df.columns:
        raise ValueError("DataFrame must contain 'band' and 'importance' columns for band importance plotting.")

    # Accept CSVs with just band,importance (no aggregation needed)
    # If band_names is None, use the order from the CSV
    if band_names is not None:
        # Ensure all bands in band_names are present, add zero if missing
        bands_in_df = set(df['band'])
        missing_bands = set(band_names) - bands_in_df
        if missing_bands:
            filler = pd.DataFrame({
                'band': list(missing_bands),
                'importance': [0] * len(missing_bands)
            })
            df = pd.concat([df, filler], ignore_index=True)
        # Set order as per band_names
        df['band'] = pd.Categorical(df['band'], categories=band_names, ordered=True)
        df = df.sort_values('band')
    else:
        # Use the order as it appears in the CSV (if a string), or as in the DataFrame
        # If multiple rows per band, aggregate
        if df['band'].duplicated().any():
            df = df.groupby('band', as_index=False)['importance'].sum()
        # Otherwise, keep as is
        # No sorting

    plt.figure(figsize=(10, 6))
    plt.bar(df['band'], df['importance'], color='skyblue')
    plt.xlabel('Band')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved band importance plot to {save_path}")
    else:
        plt.show()

def plot_time_importance(df, num_timesteps=None, title="Feature Importance by Time Step", save_path=None):
    """
    Plots a bar chart of feature importances aggregated by time step.

    Args:
        df: DataFrame with columns ['time_step', 'importance'] or at least 'time_step' and 'importance'
        num_timesteps: Optional, number of time steps to display in order
        title: Title for the plot
        save_path: Optional path to save the plot

    Usage:
        Pass a DataFrame with 'time_step' and 'importance' columns.
        If num_timesteps is provided, ensures x-axis covers all time steps from 1 to num_timesteps.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)

    if 'time_step' not in df.columns or 'importance' not in df.columns:
        raise ValueError("DataFrame must contain 'time_step' and 'importance' columns for time importance plotting.")

    # Accept CSVs with just time_step,importance (no aggregation needed)
    # If num_timesteps is not provided, use the order from the CSV
    if num_timesteps is not None:
        # Ensure all time steps present, add zero if missing
        all_steps = pd.DataFrame({'time_step': list(range(1, num_timesteps+1))})
        df = all_steps.merge(df, on='time_step', how='left').fillna(0)
    else:
        # Use the order as it appears in the CSV (if a string), or as in the DataFrame
        # If multiple rows per time_step, aggregate
        if df['time_step'].duplicated().any():
            df = df.groupby('time_step', as_index=False)['importance'].sum()
        # Otherwise, keep as is
        # No sorting

    plt.figure(figsize=(12, 6))
    plt.bar(df['time_step'], df['importance'], color='coral')
    plt.xlabel('Time Step')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(df['time_step'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved time importance plot to {save_path}")
    else:
        plt.show()


# --- New utility: Plot band-time heatmap from two CSVs ---
def plot_band_time_heatmap_from_csv(band_csv, time_csv, band_names=None, num_timesteps=None, save_path=None):
    """
    Reads band and time importance CSVs, merges them into a DataFrame with columns ['band', 'time_step', 'importance']
    (filling missing combinations with 0), and plots the band-time heatmap.

    Args:
        band_csv: Path to CSV with columns 'band', 'importance'
        time_csv: Path to CSV with columns 'time_step', 'importance'
        band_names: Optional list of band names (order)
        num_timesteps: Optional number of time steps
        save_path: Optional path to save the heatmap
    """
    # Read CSVs
    band_df = pd.read_csv(band_csv)
    time_df = pd.read_csv(time_csv)

    # Determine unique bands and timesteps
    if band_names is not None:
        bands = band_names
    else:
        # Use order from band_df as read
        bands = list(band_df['band'])
    if num_timesteps is not None:
        timesteps = list(range(1, num_timesteps+1))
    else:
        timesteps = list(time_df['time_step'])

    # Build all (band, time_step) combinations
    grid = pd.DataFrame(list(product(bands, timesteps)), columns=['band', 'time_step'])

    # Merge importance values: for each (band, time_step), use band importance * time importance
    # First, make lookup dicts
    band_imp = dict(zip(band_df['band'], band_df['importance']))
    time_imp = dict(zip(time_df['time_step'], time_df['importance']))
    grid['importance'] = [
        band_imp.get(b, 0) * time_imp.get(t, 0) for b, t in zip(grid['band'], grid['time_step'])
    ]

    # Plot using the main utility
    plot_band_time_importance(
        grid,
        band_names=bands,
        num_timesteps=len(timesteps),
        save_path=save_path
    )

def plot_feature_importance_from_df(df, band_names=None, num_timesteps=None,
                                    title_prefix="Feature Importance",
                                    save_path=None):
    """
    Utility to automatically detect the type of feature importance DataFrame and plot accordingly.

    Args:
        df: DataFrame or path to CSV containing feature importance data.
        band_names: Optional list of band names (for band or band_time plots).
        num_timesteps: Optional number of time steps (for time or band_time plots).
        title_prefix: Prefix string for plot titles.
        save_path: Optional path to save the plot as PNG.
    """
    if isinstance(df, str):
        import pandas as pd
        df = pd.read_csv(df)

    has_band = 'band' in df.columns
    has_time = 'time_step' in df.columns

    if has_band and has_time:
        plot_band_time_importance(df, band_names=band_names, num_timesteps=num_timesteps,
                                 title=f"{title_prefix} by Band and Time Step", save_path=save_path)
    elif has_band:
        plot_band_importance(df, band_names=band_names,
                             title=f"{title_prefix} by Band", save_path=save_path)
    elif has_time:
        plot_time_importance(df, num_timesteps=num_timesteps,
                             title=f"{title_prefix} by Time Step", save_path=save_path)
    else:
        raise ValueError("DataFrame must contain at least 'band' or 'time_step' column for plotting.")


# --- Helper functions for evaluation and metrics ---

def get_image_metadata(ids):
    '''
    Retrieves information about presence of a water source, month, and year for each image with 
    id in ids

    Params
        - ids (list): List of unique IDs for each pixel (n_samples, 1)
    
    Returns
        - months (np.array): List of integers between 1 and 12, where integer at element i is 
          the month of the image with unique id as ids[i]
        - years (np.array): List of integers between 2016 and 2025, where integer at element i is
          the year of the image with unique id as ids[i]
        - water_sources (np.array): List of booleans, where boolean at element i represents the 
          presence/absence of a water source in the image with unique id as ids[i]
    '''
    irrigation_table = pd.read_csv(LABEL_CSV)
    months = [irrigation_table.loc[irrigation_table['unique_id'] == i, 'month'].values[0] for i in ids]
    years = [irrigation_table.loc[irrigation_table['unique_id'] == i, 'year'].values[0] for i in ids]
    water_sources = [irrigation_table.loc[irrigation_table['unique_id'] == i, 'water_source'].values[0] for i in ids]
    months = np.array(months)
    years = np.array(years)
    water_sources = np.array(water_sources)
    return months, years, water_sources

def get_metrics(truth, pred, target_names):
    '''
    Computes the precision, recall, and F1-score for a given dimension and category.
    Returns a dictionary containing the metrics for each irrigation type

    Params
        - truth (np.array) of shape (n_samples, H, W): Ground truth labels. 
        - pred (np.array) of shape (n_samples, H, W): Predicted labels.
        - target names (list of strings): The names of the irrigation classes (eg: "Small-scale"). Should match
          with the value speciifed in label bands. (eg: Not irrigated is 0, small-scale is 1)
    '''
    # Flatten the arrays to 1D for sklearn
    y_true = truth.flatten()
    y_pred = pred.flatten()

    labels = list(range(len(target_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)

    results = {}
    for class_name in target_names:
        class_metrics = report[class_name]
        results[class_name.lower().replace(" ", "_").replace("-", "_")] = {
            'f1-score': class_metrics['f1-score'],
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'support': class_metrics['support']
        }
    return results

def get_uncertainty_explanation_metrics(label_metadata, y_pred, y_test, target_names):
    """
    Computes metrics for each uncertainty explanation category.
    
    Args:
        label_metadata (np.ndarray): Metadata array with shape (n_samples, 6, H, W).
        y_pred (np.ndarray): Predicted labels with shape (n_samples, H, W).
        y_test (np.ndarray): Ground truth labels with shape (n_samples, H, W).
    
    Returns:
        dict: Metrics for each uncertainty explanation category.
    """
    metrics = {}
    for i in range(5):
        mask = np.where(label_metadata[:, i, :, :] == 1)
        category_name = UNCERTAINTY_EXPLANATIONS[i].lower().replace(" ", "_").replace("-", "_")
        metrics[category_name] = get_metrics(y_test[mask], y_pred[mask], target_names)
    return metrics

def get_certainty_score_metrics(label_metadata, y_pred, y_test, target_names):
    """
    Computes metrics for each certainty score category.
    
    Args:
        label_metadata (np.ndarray): Metadata array with shape (n_samples, 6, H, W).
        y_pred (np.ndarray): Predicted labels with shape (n_samples, H, W).
        y_test (np.ndarray): Ground truth labels with shape (n_samples, H, W).
    
    Returns:
        dict: Metrics for each certainty score category.
    """
    metrics = {}
    certainty_scores = label_metadata[:, 5, :, :]
    
    low_mask = np.where(certainty_scores <= 3)
    metrics['low_certainty'] = get_metrics(y_test[low_mask], y_pred[low_mask], target_names)
    
    high_mask = np.where(certainty_scores > 3)
    metrics['high_certainty'] = get_metrics(y_test[high_mask], y_pred[high_mask], target_names)
    
    return metrics


def get_month_metrics(months, y_pred, y_test, target_names):
    """
    Computes metrics for each month.
    
    Args:
        months (np.ndarray): Array of months with shape (n_samples,).
        y_pred (np.ndarray): Predicted labels with shape (n_samples, H, W).
        y_test (np.ndarray): Ground truth labels with shape (n_samples, H, W).
    
    Returns:
        dict: Metrics for each month.
    """
    metrics = {}

    for i in range(6, 11):  # June to October
        mask = np.where(months == i)
        category_name = calendar.month_name[i].lower()
        metrics[category_name] = get_metrics(y_test[mask], y_pred[mask], target_names)
    return metrics


def get_year_metrics(years, y_pred, y_test, target_names):
    """
    Computes metrics for each year.
    
    Args:
        years (np.ndarray): Array of years with shape (n_samples,).
        y_pred (np.ndarray): Predicted labels with shape (n_samples, H, W).
        y_test (np.ndarray): Ground truth labels with shape (n_samples, H, W).
    
    Returns:
        dict: Metrics for each year.
    """
    metrics = {}
    for year in range(2016, 2026):
        mask = np.where(years == year)
        metrics[str(year)] = get_metrics(y_test[mask], y_pred[mask], target_names)
    return metrics

def get_water_source_metrics(water_sources, y_pred, y_test, target_names):
    """
    Computes metrics based on the presence of a water source.
    
    Args:
        water_sources (np.ndarray): Array indicating presence of water source with shape (n_samples,).
        y_pred (np.ndarray): Predicted labels with shape (n_samples, H, W).
        y_test (np.ndarray): Ground truth labels with shape (n_samples, H, W).
    
    Returns:
        dict: Metrics for presence and absence of water source.
    """
    metrics = {}
    
    # Presence of water source
    presence_mask = np.where(water_sources == True)
    metrics['water_source_present'] = get_metrics(y_test[presence_mask], y_pred[presence_mask], target_names)
    
    # Absence of water source
    absence_mask = np.where(water_sources == False)
    metrics['water_source_absent'] = get_metrics(y_test[absence_mask], y_pred[absence_mask], target_names)
    
    return metrics

def get_class_presence(mask, num_classes=2, presence_thresh=1):
    """
    Returns a binary array of class presence for a single image, where each element 
    corresponds to a separate class.

    Params:
        - mask: The image over which we want to determine irrigation presence per-class
        - num_classes (int): The number of irrigation classes (2 for binary, 6 for multi-class)
        - presence_thresh (int): Threshold for how many pixels must be categorized with
          a particular class, for the class to be considered present in the image.

    Returns:
        - np.array of shape (num_classes,), where each element is 0 or 1, where 0 
          represents the absence of class i in the image, and 1 represents the presence
          of class i in the image.
    """
    return np.array([
        int(np.sum(mask == c) >= presence_thresh)
        for c in range(num_classes)
    ])
    
def compute_presence_metrics(preds, gts, target_names, presence_thresh=1):
    """
    Uses sklearn to compute multi-label class presence metrics.
    
    Args:
        preds (list): List of predicted label images. Shape (n_images, H, W)
        gts (list): List of ground truth label images. Shape (n_images, H, W)
        target_names (list of strings): List of irrigation category names
        presence_thresh: Minimum pixel count to consider a class present.
        average: 'macro', 'micro', 'samples', or 'weighted' — standard sklearn averaging.
    
    Returns:
        Dictionary with sklearn-style precision, recall, F1.
    """
    # (n_samples, n_classes)
    num_classes = len(target_names)
    Y_pred = np.array([get_class_presence(p, num_classes, presence_thresh) for p in preds])
    Y_true = np.array([get_class_presence(g, num_classes, presence_thresh) for g in gts])
    per_class_metrics = {}

    for idx in range(0, num_classes):
        pred = Y_pred[:,idx]
        true = Y_true[:,idx]
        category = target_names[idx].lower().replace(" ", "_").replace("-", "_")
        per_class_metrics[category] = {
                    "precision": precision_score(true, pred, zero_division=0),
                    "recall": recall_score(true, pred, zero_division=0),
                    "f1-score": f1_score(true, pred, zero_division=0),
                    }
    return per_class_metrics

def metrics_over_factors(y_pred, y_test, multi_class, label_metadata, ids, metrics_path):
    '''
    Assesses the pixel-level metrics for the model. Retrieves F1 score, accuracy, and recall for
    each category in each dimension, allowing for easy analysis of how the model performs across
    different categories.

    For example, for dimension 'uncertainty explanation', there are multiple categories, such 
    as "uneven" or "only slightly green." For each category, we retrieve F1-score, accuracy,
    and recall, such that we can determine if the model performs worse for a particular category.

    Returns a list of dictionaries containing the metrics for each factor.

    Parameters
        - y_pred (np.arr): List of predictions, shape (n_samples, H, W)
        - y_test (np.arr): List of ground truth values (n_samples, H, W)
        - multi_class (bool): Whether we performed multi-class classification or binary classification.
        - label_metadata (np.arr): List of associated metadata per pixel (n_samples, 6, H, W)
        - ids (np.array): List of unique IDs for each pixel (n_samples,)
        - metrics_path (str): Path to save the metrics JSON file.

    Returns:
        - dict with structure:
            - pixel_metrics:
                - overall: {irrigation_class: {precision, recall, f1-score, support}}
                - per_uncertainty_explanation: {explanation: {irrigation_class: {metrics}}}
                - per_uncertainty_score: {score_bin: {irrigation_class: {metrics}}}
                - per_month / per_year / water_source: same structure as above
            - image_metrics:
                - image_level_class_presence: {irrigation_class: {metrics}}
                - image_level_fraction_irrigated: {mae, rmse, mse}
    '''
    # Check that y_pred, y_test, label_metadata, and ids have the same number of samples
    assert y_pred.shape == y_test.shape, "y_pred and y_test must have the same shape"
    assert y_pred.shape[0] == label_metadata.shape[0],  "y_pred and label_metadata must have the same number of samples"
    assert y_pred.shape[0] == len(ids), "y_pred and ids must have the same number of samples"
    assert label_metadata.shape[1] == 6, "label_metadata must have shape (n_samples, 6, H, W)"
    assert label_metadata.shape[2:] == y_pred.shape[1:], "label_metadata and y_pred must have the same H, W dimensions"

    pixel_metrics = {} # pixel-level metrics
    image_metrics = {} # image-level metrics

    # Retrieve information about dates (month, year) and presence of water source for each pixel in the image
    months, years, water_sources = get_image_metadata(ids)

    # F1 and accuracy per-class
    target_names = BINARY_CLASSES
    if multi_class:
        target_names = MULTI_CLASSES

    pixel_metrics['overall'] = get_metrics(y_test, y_pred, target_names)
    pixel_metrics['per_uncertainty_explanation'] = get_uncertainty_explanation_metrics(label_metadata, y_pred, y_test, target_names)
    pixel_metrics['per_uncertainty_score'] = get_certainty_score_metrics(label_metadata, y_pred, y_test, target_names)
    pixel_metrics['per_month'] = get_month_metrics(months, y_pred, y_test, target_names)
    pixel_metrics['per_year'] = get_year_metrics(years, y_pred, y_test, target_names)
    pixel_metrics['water_source'] = get_water_source_metrics(water_sources, y_pred, y_test, target_names)

    # Todo: Water source metrics

    # Metrics for image-level class detection
    image_metrics['image_level_class_presence'] = compute_presence_metrics(y_pred, y_test, target_names)

    # Determine how much of each image is irrigation detected in, report MAE/RMSE/MSE
    pred_fractions = []
    true_fractions = []
    for idx in range(y_pred.shape[0]):
        pred_img = y_pred[idx]
        truth_img = y_test[idx]
        pred_fractions.append(np.mean(pred_img > 0))
        true_fractions.append(np.mean(truth_img > 0))

    image_metrics['image_level_fraction_irrigated'] = {
        'mae': mean_absolute_error(true_fractions, pred_fractions),
        'rmse': mean_squared_error(true_fractions, pred_fractions),
        'mse': root_mean_squared_error(true_fractions, pred_fractions)
    }

    # Return metrics, save to JSON file
    metrics = {}
    metrics['pixel_metrics'] = pixel_metrics
    metrics['image_metrics'] = image_metrics

    # Dump metrics into a JSON file
    file_path = os.path.join(metrics_path, "metrics.json")
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def plot_metrics_over_factors(metrics_json, save_dir="plots"):
    '''
    Plots the metrics over different factors and saves the figures to a folder.

    Parameters:
        - metrics_json (dict): Dictionary output from metrics_over_factors
        - save_dir (str): Path to directory where plots should be saved
    '''

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    pixel_metrics = metrics_json["pixel_metrics"]
    image_metrics = metrics_json["image_metrics"]
    irrigation_classes = BINARY_CLASSES
    if len(metrics_json['pixel_metrics']['overall'].keys()) == 6:
        irrigation_classes = MULTI_CLASSES

    def extract_data(section_key, image_level):
        '''
        For a specific factor, return a pd.DataFrame, where each entry corresponds
        to a particular category. This is then used to plot the data.

        Parameters
            - section_key (string): A specific factor (eg: uncertainty explanation)
            - image_level (bool): True we are performing image-level analysis, false if it is a pixel-level analysis
        Returns
            - pd.DataFrame, with each entry corresponding metrics for a particular category for each class
                - Example of a row: {'category': 'Unclear signs of agriculture', 'class': 'small_scale', ...scores}
        '''
        section_data = image_metrics[section_key] if image_level else pixel_metrics[section_key]
        data = []
        if section_key == 'overall' or image_level:
           for irrigation_class in irrigation_classes:
                # Key to access irrigation class from dict
                irrigation_class_key = irrigation_class.lower().replace(" ", "_").replace("-", "_")
                metrics = section_data.get(irrigation_class_key, {})
                data.append({
                    "category": "",
                    "class": irrigation_class, 
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1-score": metrics.get("f1-score", 0.0),
                })
        else:
            for category, category_data in section_data.items():
                for irrigation_class in irrigation_classes:
                    # Key to access irrigation class from dict
                    irrigation_class_key = irrigation_class.lower().replace(" ", "_").replace("-", "_")
                    metrics = category_data.get(irrigation_class_key, {})
                    data.append({
                        "category": category.replace("_", " ").capitalize(),
                        "class": irrigation_class, 
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "f1-score": metrics.get("f1-score", 0.0),
                        "support": metrics.get("support", 0.0)
                    })
        return pd.DataFrame(data)

    def make_plot(df, metric, title, filename):
        # Do not plot categories with 0 cases
        if "support" in df.columns.tolist():
            df = df[df["support"] > 0]
            if df.empty:
                print(f"Skipping plot for {title} ({metric}) — all entries have zero support.")
                return
        plt.figure(figsize=(12, 5))
        ax = plt.subplot()
        pivot_df = df.pivot(index='category', columns='class', values=metric.lower())
        pivot_df.plot(kind='bar', ax=ax, width=0.85)

        # Different plot title for overall and image-level class presence graphs
        plot_title = f"{metric} per {title}"
        x_label = 'Category'
        if title == 'Overall':
            plot_title = f"Overall {metric}"
            x_label = ""
        elif title == 'Image-Level Class Presence':
            plot_title = f"{metric} for Image-Level Class Presence Detection"
            x_label = ""

        plt.title(plot_title)
        plt.ylabel(metric)
        plt.xlabel(x_label)
        plt.xticks(rotation=15)
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{filename}_{metric.lower()}.png")
        plt.savefig(plot_path)
        plt.close()

    # Handle one or more grouping sections
    sections_to_plot = {
        "overall": "Overall",
        "per_uncertainty_explanation": "Uncertainty Explanation",
        "per_uncertainty_score": "Uncertainty Score",
        "per_month": "Month",
        "per_year": "Year",
        "water_source": "Water Source",
        "image_level_class_presence": "Image-Level Class Presence"
    }

    # Plot precision, recall, f1-score (3 plots) 
    for section_key, title in sections_to_plot.items():
        image_level = False
        if section_key == 'image_level_class_presence':
            image_level = True
        df = extract_data(section_key, image_level)
        for metric in ['Precision', 'Recall', 'F1-Score']:
            make_plot(df, metric, title, section_key)
    '''
    Plots the metrics over different factors and saves the figures to a folder.

    Parameters:
        - metrics_json (dict): Dictionary output from metrics_over_factors
        - save_dir (str): Path to directory where plots should be saved
    '''

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    pixel_metrics = metrics_json["pixel_metrics"]
    image_metrics = metrics_json["image_metrics"]
    irrigation_classes = BINARY_CLASSES
    if len(metrics_json['pixel_metrics']['overall'].keys()) == 6:
        irrigation_classes = MULTI_CLASSES

    def extract_data(section_key, image_level):
        '''
        For a specific factor, return a pd.DataFrame, where each entry corresponds
        to a particular category. This is then used to plot the data.

        Parameters
            - section_key (string): A specific factor (eg: uncertainty explanation)
            - image_level (bool): True we are performing image-level analysis, false if it is a pixel-level analysis
        Returns
            - pd.DataFrame, with each entry corresponding metrics for a particular category for each class
                - Example of a row: {'category': 'Unclear signs of agriculture', 'class': 'small_scale', ...scores}
        '''
        section_data = image_metrics[section_key] if image_level else pixel_metrics[section_key]
        data = []
        if section_key == 'overall' or image_level:
           for irrigation_class in irrigation_classes:
                # Key to access irrigation class from dict
                irrigation_class_key = irrigation_class.lower().replace(" ", "_").replace("-", "_")
                metrics = section_data.get(irrigation_class_key, {})
                data.append({
                    "category": "",
                    "class": irrigation_class, 
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1-score": metrics.get("f1-score", 0.0),
                })
        else:
            for category, category_data in section_data.items():
                for irrigation_class in irrigation_classes:
                    # Key to access irrigation class from dict
                    irrigation_class_key = irrigation_class.lower().replace(" ", "_").replace("-", "_")
                    metrics = category_data.get(irrigation_class_key, {})
                    data.append({
                        "category": category.replace("_", " ").capitalize(),
                        "class": irrigation_class, 
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "f1-score": metrics.get("f1-score", 0.0),
                        "support": metrics.get("support", 0.0)
                    })
        return pd.DataFrame(data)

    def make_plot(df, metric, title, filename):
        # Do not plot categories with 0 cases
        if "support" in df.columns.tolist():
            df = df[df["support"] > 0]
            if df.empty:
                print(f"Skipping plot for {title} ({metric}) — all entries have zero support.")
                return
        plt.figure(figsize=(12, 5))
        ax = plt.subplot()
        pivot_df = df.pivot(index='category', columns='class', values=metric.lower())
        pivot_df.plot(kind='bar', ax=ax, width=0.85)

        # Different plot title for overall and image-level class presence graphs
        plot_title = f"{metric} per {title}"
        x_label = 'Category'
        if title == 'Overall':
            plot_title = f"Overall {metric}"
            x_label = ""
        elif title == 'Image-Level Class Presence':
            plot_title = f"{metric} for Image-Level Class Presence Detection"
            x_label = ""

        plt.title(plot_title)
        plt.ylabel(metric)
        plt.xlabel(x_label)
        plt.xticks(rotation=15)
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{filename}_{metric.lower()}.png")
        plt.savefig(plot_path)
        plt.close()

    # Handle one or more grouping sections
    sections_to_plot = {
        "overall": "Overall",
        "per_uncertainty_explanation": "Uncertainty Explanation",
        "per_uncertainty_score": "Uncertainty Score",
        "per_month": "Month",
        "per_year": "Year",
        "water_source": "Water Source",
        "image_level_class_presence": "Image-Level Class Presence"
    }

    # Plot precision, recall, f1-score (3 plots) 
    for section_key, title in sections_to_plot.items():
        image_level = False
        if section_key == 'image_level_class_presence':
            image_level = True
        df = extract_data(section_key, image_level)
        for metric in ['Precision', 'Recall', 'F1-Score']:
            make_plot(df, metric, title, section_key)