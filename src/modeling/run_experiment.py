import os
import sys
import yaml
import json
import shutil
import logging
from datetime import datetime
from joblib import dump
from ml_pipeline.ml_model import train_model
from ml_pipeline.evaluation import model_metrics
from ml_pipeline.visualization import plot_ml_predictions
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

    # Resolve paths
    base_dir = exp_cfg["output"]["base_dir"]
    experiment_dir = os.path.join(base_dir, run_name)

    # Skip if directory already exists (prevent duplicates)
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

    # Log file setup
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            logger.info(f"[{timestamp}] Starting experiment: {base_name}")
            logger.info(f"Saving outputs to: {experiment_dir}")

            # Prepare data splits
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
                    with open(split_metadata_path, "w") as f:
                        json.dump(split_metadata, f, indent=2, default=str)
                    logger.info(f"Split metadata saved to: {split_metadata_path}")
                    
            else:
                logger.info("Using manual file lists from config...")
                # Use existing hardcoded lists
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

            from ml_pipeline.build_features import flatten_dataset
            X_train, y_train = flatten_dataset(train_dataset)
            X_val, y_val = flatten_dataset(val_dataset)

            logger.info(f"Flattened data shapes:")
            logger.info(f"  - X_train: {X_train.shape}")
            logger.info(f"  - y_train: {y_train.shape}")
            logger.info(f"  - X_val: {X_val.shape}")
            logger.info(f"  - y_val: {y_val.shape}")

            # Select only first two label bands for ML training/validation
            # This restricts ML training to the first two bands, reserving other bands for post-hoc analysis
            y_train = y_train[:, :2]
            y_val = y_val[:, :2]

            # Train model
            model_type = exp_cfg["model"]["type"].lower()
            hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {}) # get the hyperparameters specific to this model

            logger.info(f"Training {model_type} model with hyperparameters: {hyperparams}")
            clf = train_model(X_train, y_train, model_type, **hyperparams)

            dump(clf, model_path) # Warning -- with deep learning models we will want to do this in epochs
            logger.info(f"Model saved to {model_path}")

            # Predict and evaluate
            y_pred = clf.predict(X_val)

            #reshape y_pred to original tensor shape for post-hoc 
            # sample = val_dataset[0]
            # H, W = sample['mask'].shape[-2:]

            # mask = sample['mask']
            # if mask.ndim == 2:
            #     mask_flat = mask.reshape(H * W)
            #     valid_mask = mask_flat != -1
            # elif mask.ndim == 3:
            #     mask_flat = mask.permute(1, 2, 0).reshape(H * W, mask.shape[0])
            #     valid_mask = ~np.any(mask_flat == -1, axis=1)
            # else:
            #     raise ValueError("Unexpected mask shape.")

            # full_pred = np.full((H * W, 2), fill_value=-1, dtype=y_pred.dtype)
            # full_pred[valid_mask] = y_pred
            # y_pred_reshaped = full_pred.reshape(H, W, 2)

            metrics = model_metrics(y_pred, y_val)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Metrics:", metrics)

            # Visualization
            num_samples = exp_cfg["visualization"].get("num_samples", 2)

            plot_ml_predictions(
                val_dataset, clf,
                num_samples=num_samples, save_path=visualization_path
            )
            logger.info(f"[{timestamp}] Experiment complete.")

        finally:
            sys.stdout = original_stdout
            logger.info(f"Logged output to {log_path}")


if __name__ == "__main__":
    config_path = "experiment.yaml"
    experiments = load_experiment(config_path)
    # If the config is a list of experiments, iterate; else, wrap in a list
    if isinstance(experiments, list):
        for exp_cfg in experiments:
            run_experiment(exp_cfg, config_path)
    else:
        run_experiment(experiments, config_path)