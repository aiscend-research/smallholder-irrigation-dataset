from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os

def model_metrics(y_pred, y_test):
    """
    Computes accuracy, precision, recall, and F1 score for the first two mask bands:
        - Band 1: Irrigation Type (0–5, categorical)
        - Band 2: Irrigation Presence (0/1, binary)
    Args:
        y_pred: numpy array of shape (N, >=2)
        y_test: numpy array of shape (N, >=2)
    Returns:
        dict: Dictionary with per-band metrics.
    """
    metrics = {}

    # Ensure 2D shape
    if y_pred.ndim == 1 or y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # Only use the first two bands
    y_pred = y_pred[:, :2]
    y_test = y_test[:, :2]

    # Band 1: Irrigation type (categorical: 0–5)
    acc1 = accuracy_score(y_test[:, 0], y_pred[:, 0])
    prec1 = precision_score(y_test[:, 0], y_pred[:, 0], average='weighted', zero_division=0)
    rec1 = recall_score(y_test[:, 0], y_pred[:, 0], average='weighted', zero_division=0)
    f1_1 = f1_score(y_test[:, 0], y_pred[:, 0], average='weighted', zero_division=0)
    metrics["irrigation_type"] = {
        "accuracy": acc1,
        "precision": prec1,
        "recall": rec1,
        "f1_score": f1_1
    }

    # Band 2: Irrigation presence (binary: 0/1)
    acc2 = accuracy_score(y_test[:, 1], y_pred[:, 1])
    prec2 = precision_score(y_test[:, 1], y_pred[:, 1], average='binary', zero_division=0)
    rec2 = recall_score(y_test[:, 1], y_pred[:, 1], average='binary', zero_division=0)
    f1_2 = f1_score(y_test[:, 1], y_pred[:, 1], average='binary', zero_division=0)
    metrics["irrigation_presence"] = {
        "accuracy": acc2,
        "precision": prec2,
        "recall": rec2,
        "f1_score": f1_2
    }

    return metrics


# --- Feature importance export utility ---
def export_feature_importances(clf, band_names, num_timesteps, out_dir="./", prefix=""):
    """
    Exports feature importances as a DataFrame and saves:
      - Detailed importance (band, timestep)
      - Aggregated by band
      - Aggregated by time_step
    Args:
        clf: Trained MultiOutputClassifier wrapping RandomForest or GradientBoosting
        band_names: list of strings, length = num_bands
        num_timesteps: int, number of time points (e.g., 37)
        out_dir: where to save csvs
        prefix: optional, filename prefix
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Get base estimator (works for MultiOutputClassifier)
    if hasattr(clf, "estimators_"):
        # Multioutput: average importances across outputs
        all_importances = np.array([est.feature_importances_ for est in clf.estimators_])
        importances = all_importances.mean(axis=0)
    else:
        importances = clf.feature_importances_
    
    num_bands = len(band_names)
    feature_names = []
    for t in range(num_timesteps):
        for b in range(num_bands):
            feature_names.append(f"{band_names[b]}_t{t+1}")

    # --- NEW: Ensure feature_names matches importances in length ---
    if len(feature_names) != len(importances):
        print(f"[WARNING] Mismatch in feature_names ({len(feature_names)}) and importances ({len(importances)}); truncating to match shorter length.")
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    # Extract band and timestep for grouping
    df["band"] = df["feature"].str.extract(r"^(.*?)_t")[0]
    df["time_step"] = df["feature"].str.extract(r"_t(\d+)$")[0].astype(int)

    # Aggregate
    agg_band = df.groupby("band")["importance"].sum().sort_values(ascending=False).reset_index()
    agg_time = df.groupby("time_step")["importance"].sum().sort_values(ascending=False).reset_index()

    # Save to CSV
    df.to_csv(os.path.join(out_dir, f"{prefix}feature_importance_detailed.csv"), index=False)
    agg_band.to_csv(os.path.join(out_dir, f"{prefix}feature_importance_by_band.csv"), index=False)
    agg_time.to_csv(os.path.join(out_dir, f"{prefix}feature_importance_by_time.csv"), index=False)
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

    agg = df.groupby('band')['importance'].sum().reset_index()
    if band_names:
        # Ensure all band_names are present, fill missing with 0 importance
        missing_bands = set(band_names) - set(agg['band'])
        if missing_bands:
            for mb in missing_bands:
                agg = agg.append({'band': mb, 'importance': 0}, ignore_index=True)
        agg['band'] = pd.Categorical(agg['band'], categories=band_names, ordered=True)
        agg = agg.sort_values('band')

    plt.figure(figsize=(10, 6))
    plt.bar(agg['band'], agg['importance'], color='skyblue')
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

    agg = df.groupby('time_step')['importance'].sum().reset_index()
    if num_timesteps:
        all_steps = pd.DataFrame({'time_step': list(range(1, num_timesteps+1))})
        agg = all_steps.merge(agg, on='time_step', how='left').fillna(0)

    plt.figure(figsize=(12, 6))
    plt.bar(agg['time_step'], agg['importance'], color='coral')
    plt.xlabel('Time Step')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(agg['time_step'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved time importance plot to {save_path}")
    else:
        plt.show()

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
