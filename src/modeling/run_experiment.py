import os
import sys
import yaml
import json
import shutil
import logging
import numpy as np
from datetime import datetime
from joblib import dump
from ml_pipeline.ml_model import train_model
from ml_pipeline.evaluation import model_metrics
from ml_pipeline.visualization import plot_ml_predictions
from ml_pipeline.build_features import flatten_dataset  # ← Moved to top
from custom_dataset import MultiTemporalCropDataset
from ml_pipeline.data_splitting import IrrigationDataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_experiment(config_path="experiment.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(exp_cfg, config_path):
    # Timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = exp_cfg["name"]
    run_name = f"{base_name}_{timestamp}"

    base_dir = exp_cfg["output"]["base_dir"]
    experiment_dir = os.path.join(base_dir, run_name)

    if os.path.exists(experiment_dir):
        logger.info(f"Skipping: {run_name} already exists.")
        return

    os.makedirs(experiment_dir, exist_ok=True)

    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    visualization_path = os.path.join(experiment_dir, "visualization.png")
    config_snapshot_path = os.path.join(experiment_dir, "experiment.yaml")
    log_path = os.path.join(experiment_dir, "run.log")
    split_metadata_path = os.path.join(experiment_dir, "split_metadata.json")

    # Copy the config file used for this experiment
    shutil.copyfile(config_path, config_snapshot_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            logger.info(f"[{timestamp}] Starting experiment: {base_name}")
            logger.info(f"Saving outputs to: {experiment_dir}")

            # Check if cross-validation is enabled
            use_cross_validation = exp_cfg["data"].get("use_cross_validation", False)
            
            if use_cross_validation:
                logger.info("Running cross-validation experiment...")
                return run_cv_experiment(exp_cfg, experiment_dir)
            else:
                logger.info("Running single train/val experiment...")
                return run_single_experiment(exp_cfg, experiment_dir)

        finally:
            sys.stdout = original_stdout
            logger.info(f"Logged output to {log_path}")


def run_single_experiment(exp_cfg, experiment_dir):
    """Run a single train/validation experiment."""
    use_auto_splitting = exp_cfg["data"].get("use_auto_splitting", True)
    
    if use_auto_splitting:
        logger.info("Using automatic data splitting...")
        
        # Initialize data splitter
        splitter = IrrigationDataSplitter(
            csv_path=exp_cfg["data"]["csv_path"],
            data_dir=exp_cfg["data"]["data_dir"],
            random_state=exp_cfg["data"].get("random_state", 42)
        )
        
        # Use the splitter's integration method
        train_files, val_files, split_metadata = splitter.prepare_experiment_splits(
            exp_cfg, experiment_dir
        )
        
        # Save split metadata
        if split_metadata:
            split_metadata_path = os.path.join(experiment_dir, "split_metadata.json")
            with open(split_metadata_path, "w") as f:
                json.dump(split_metadata, f, indent=2, default=str)
            logger.info(f"Split metadata saved to: {split_metadata_path}")
            
    else:
        logger.info("Using manual file lists from config...")
        train_files = exp_cfg["data"]["train_files"]
        val_files = exp_cfg["data"]["val_files"]
        split_metadata = None

    # Prepare data
    data_dir = exp_cfg["data"]["data_dir"]
    label_bands = exp_cfg["data"]["label_bands"]

    logger.info(f"Creating datasets:")
    logger.info(f"  - Data directory: {data_dir}")
    logger.info(f"  - Train files: {len(train_files)}")
    logger.info(f"  - Val files: {len(val_files)}")
    logger.info(f"  - Label bands: {label_bands}")

    train_dataset = MultiTemporalCropDataset(
        data_dir=data_dir,
        sample_file_list=train_files,
        label_bands=label_bands
    )
    val_dataset = MultiTemporalCropDataset(
        data_dir=data_dir,
        sample_file_list=val_files,
        label_bands=label_bands
    )

    logger.info(f"Dataset sizes:")
    logger.info(f"  - Train dataset: {len(train_dataset)} samples")
    logger.info(f"  - Val dataset: {len(val_dataset)} samples")

    X_train, y_train = flatten_dataset(train_dataset)
    X_val, y_val = flatten_dataset(val_dataset)

    logger.info(f"Flattened data shapes:")
    logger.info(f"  - X_train: {X_train.shape}")
    logger.info(f"  - y_train: {y_train.shape}")
    logger.info(f"  - X_val: {X_val.shape}")
    logger.info(f"  - y_val: {y_val.shape}")

    # Select only first two label bands for ML training/validation
    y_train = y_train[:, :2]
    y_val = y_val[:, :2]

    # Train model
    model_type = exp_cfg["model"]["type"].lower()
    hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})

    logger.info(f"Training {model_type} model with hyperparameters: {hyperparams}")
    clf = train_model(X_train, y_train, model_type, **hyperparams)

    model_path = os.path.join(experiment_dir, "model.pkl")
    dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    # Predict and evaluate
    y_pred = clf.predict(X_val)
    metrics = model_metrics(y_pred, y_val)
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics:", metrics)

    num_samples = exp_cfg["visualization"].get("num_samples", 2)
    visualization_path = os.path.join(experiment_dir, "visualization.png")

    plot_ml_predictions(
        val_dataset, clf,
        num_samples=num_samples, save_path=visualization_path
    )
    logger.info(f"Experiment complete.")


def run_cv_experiment(exp_cfg, experiment_dir):
    """Run cross-validation experiment."""
    # Initialize data splitter
    splitter = IrrigationDataSplitter(
        csv_path=exp_cfg["data"]["csv_path"],
        data_dir=exp_cfg["data"]["data_dir"],
        random_state=exp_cfg["data"].get("random_state", 42)
    )
    
    # Get CV parameters
    n_folds = exp_cfg["data"].get("n_folds", 5)
    cv_structure_name = exp_cfg["data"].get("cv_structure_name", "irrigation_cv")
    splits_dir = exp_cfg["data"].get("splits_dir", "./splits")
    
    # Create CV folder structure
    cv_dir = splitter.create_cv_folder_structure(
        n_splits=n_folds,
        output_dir=splits_dir,
        structure_name=cv_structure_name,
        copy_files=False  # Use file lists for memory efficiency
    )
    logger.info(f"CV structure created at: {cv_dir}")
    
    # Prepare data processing parameters
    data_dir = exp_cfg["data"]["data_dir"]
    label_bands = exp_cfg["data"]["label_bands"]
    model_type = exp_cfg["model"]["type"].lower()
    hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})
    
    # Run experiments on each fold
    logger.info(f"Running experiments on {n_folds} folds...")
    fold_results = []
    
    for fold_idx in range(1, n_folds + 1):
        logger.info(f"\n--- Fold {fold_idx} ---")
        
        # Load fold file lists
        fold_dir = os.path.join(cv_dir, "train", f"fold_{fold_idx}")
        train_file_path = os.path.join(fold_dir, "inner_train", "train_files.txt")
        val_file_path = os.path.join(fold_dir, "inner_val", "val_files.txt")
        
        # Check if fold directory exists
        if not os.path.exists(fold_dir):
            logger.warning(f"Fold {fold_idx} directory not found, skipping...")
            continue
            
        # Load train files
        if not os.path.exists(train_file_path):
            logger.warning(f"Train files not found for fold {fold_idx}, skipping...")
            continue
            
        with open(train_file_path, 'r') as f:
            train_files = [line.strip() for line in f.readlines()]
        
        # Load val files (might be empty for small datasets)
        val_files = []
        if os.path.exists(val_file_path):
            with open(val_file_path, 'r') as f:
                val_files = [line.strip() for line in f.readlines()]
        
        logger.info(f"Fold {fold_idx}: {len(train_files)} train, {len(val_files)} val")
        
        # Create datasets
        train_dataset = MultiTemporalCropDataset(
            data_dir=data_dir,
            sample_file_list=train_files,
            label_bands=label_bands
        )
        
        if len(train_dataset) == 0:
            logger.error(f"Fold {fold_idx}: Train dataset is empty!")
            continue
            
        val_dataset = MultiTemporalCropDataset(
            data_dir=data_dir,
            sample_file_list=val_files,
            label_bands=label_bands
        )
        
        if len(val_dataset) == 0:
            logger.warning(f"Fold {fold_idx}: Val dataset is empty, skipping evaluation")
            continue
        
        # Flatten datasets
        X_train, y_train = flatten_dataset(train_dataset)
        X_val, y_val = flatten_dataset(val_dataset)
        
        # Select only first two label bands
        y_train = y_train[:, :2]
        y_val = y_val[:, :2]
        
        # Train model
        logger.info(f"Training {model_type} model for fold {fold_idx}")
        clf = train_model(X_train, y_train, model_type, **hyperparams)
        
        # Evaluate model
        y_pred = clf.predict(X_val)
        metrics = model_metrics(y_pred, y_val)
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'metrics': metrics,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        })
        
        logger.info(f"Fold {fold_idx} metrics: {metrics}")
    
    # Aggregate results
    if fold_results:
        # Calculate mean and std of metrics across folds
        all_metrics = {}
        for metric_name in fold_results[0]['metrics'].keys():
            values = [result['metrics'][metric_name] for result in fold_results]
            all_metrics[f"{metric_name}_mean"] = np.mean(values)
            all_metrics[f"{metric_name}_std"] = np.std(values)
        
        # Add fold details
        all_metrics['n_folds_completed'] = len(fold_results)
        all_metrics['fold_details'] = fold_results
        
        # Save CV results
        cv_results_path = os.path.join(experiment_dir, "cv_results.json")
        with open(cv_results_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        logger.info(f"CV experiment complete. Results saved to {cv_results_path}")
        logger.info(f"Mean metrics across {len(fold_results)} folds:")
        for key, value in all_metrics.items():
            if key.endswith('_mean'):
                logger.info(f"  {key}: {value:.4f}")
    else:
        logger.error("No folds completed successfully!")


if __name__ == "__main__":
    config_path = "experiment.yaml"
    experiments = load_experiment(config_path)
    # If the config is a list of experiments, iterate; else, wrap in a list
    if isinstance(experiments, list):
        for exp_cfg in experiments:
            run_experiment(exp_cfg, config_path)
    else:
        run_experiment(experiments, config_path)