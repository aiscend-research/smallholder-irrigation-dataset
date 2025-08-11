import os
import sys
import yaml
import json
import shutil
import logging
import numpy as np
from datetime import datetime
from joblib import dump
from pathlib import Path

# Add the workspace root to Python path for imports
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from src.modeling.ml_pipeline.ml_model import train_model
from src.modeling.ml_pipeline.evaluation import model_metrics
from src.modeling.ml_pipeline.evaluation import export_feature_importances
from src.modeling.ml_pipeline.evaluation import plot_band_time_importance
from src.modeling.ml_pipeline.visualization import plot_ml_predictions
from src.modeling.ml_pipeline.build_features import flatten_dataset  # ← Moved to top
from src.modeling.custom_dataset import MultiTemporalCropDataset
from src.modeling.ml_pipeline.data_splitting import IrrigationDataSplitter

# Configure logging (console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_workspace_root():
    """Get the workspace root directory by looking for common project files"""
    # Start from the directory where this script is located
    script_dir = Path(__file__).parent
    current_dir = script_dir

    # Look for common project root indicators
    while current_dir != current_dir.parent:
        # Check for multiple indicators to be more robust
        has_readme = (current_dir / "README.md").exists()
        has_requirements = (current_dir / "requirements.txt").exists()
        has_git = (current_dir / ".git").exists()
        has_config = (current_dir / "config.yaml").exists()
        
        # If we find multiple indicators, this is likely the root
        indicators_found = sum([has_readme, has_requirements, has_git, has_config])
        
        if indicators_found >= 2:  # Require at least 2 indicators
            return current_dir
            
        current_dir = current_dir.parent

    # Fallback: go up two levels from script_dir to reach workspace root
    fallback_root = script_dir.parent.parent
    return fallback_root


def resolve_path(path_str: str, base_dir: str = None) -> str:
    """Resolve a path relative to the workspace root or specified base directory"""
    if base_dir is None:
        base_dir = get_workspace_root()

    # If path is already absolute, return as-is
    if os.path.isabs(path_str):
        return path_str

    # Resolve relative to base directory
    resolved_path = os.path.join(base_dir, path_str)
    final_path = os.path.normpath(resolved_path)
    return final_path


def load_experiment(config_path="experiment.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _clean_base_id(name: str) -> str:
    """
    Derive a base id from a filename stem by stripping common suffixes.
    Enhanced to handle multiple naming patterns found in the dataset.
    
    Examples:
      'ABC_20210301_image' -> 'ABC_20210301'
      'ABC-20210301-label' -> 'ABC-20210301'
      'site_-15.88_27.74_2021_53' -> 'site_-15.88_27.74_2021_53'
      'ABC_20210301' -> 'ABC_20210301' (already clean)
    """
    stem = Path(name).stem
    
    # Remove common suffixes in order of specificity
    suffixes_to_remove = [
        "_image", "-image", 
        "_label", "-label",
        "_json", "-json"
    ]
    
    for suffix in suffixes_to_remove:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    
    return stem


def standardize_file_names(data_dir: str, csv_path: str = None, out_subdir: str = "_standardized", copy_mode: str = "copy"):
    """
    Scan data_dir for raw .tif/.json files; copy them into <data_dir>/_standardized
    using canonical names: <BASE>_image.tif, <BASE>_label.tif, <BASE>_image.json.
    
    Enhanced to handle mixed naming patterns and incomplete file sets.
    
    Returns:
        standardized_dir (str), report (dict)
    """
    from pathlib import Path
    import shutil

    data_path = Path(data_dir)
    out_dir = data_path / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index files with more flexible pattern matching
    tifs = list(data_path.glob("*.tif"))
    jsons = list(data_path.glob("*.json"))

    # More robust image/label classification
    images = []
    labels = []
    
    for p in tifs:
        name_lower = p.name.lower()
        if "label" in name_lower:
            labels.append(p)
        else:
            # Consider any .tif without "label" as an image
            images.append(p)

    # Group by cleaned base id with more flexible matching
    by_base = {}
    
    # Process images first
    for p in images:
        base = _clean_base_id(p.name)
        by_base.setdefault(base, {})["image"] = p
    
    # Process labels
    for p in labels:
        base = _clean_base_id(p.name)
        by_base.setdefault(base, {})["label"] = p
    
    # Process JSONs
    for p in jsons:
        base = _clean_base_id(p.name)
        by_base.setdefault(base, {})["json"] = p

    copied = 0
    triples = 0
    incomplete_sets = 0
    issues = []

    for base, parts in by_base.items():
        img = parts.get("image")
        lab = parts.get("label")
        jsn = parts.get("json")

        if not img or not lab or not jsn:
            incomplete_sets += 1
            missing_parts = []
            if not img: missing_parts.append("image")
            if not lab: missing_parts.append("label") 
            if not jsn: missing_parts.append("json")
            
            issues.append(
                f"Incomplete set for base '{base}': missing {', '.join(missing_parts)}"
            )
            continue

        # Create standardized names
        dst_img = out_dir / f"{base}_image.tif"
        dst_lab = out_dir / f"{base}_label.tif"
        dst_jsn = out_dir / f"{base}_image.json"

        # Copy files if they don't exist
        for src, dst in [(img, dst_img), (lab, dst_lab), (jsn, dst_jsn)]:
            if not dst.exists():
                if copy_mode == "copy":
                    shutil.copy2(src, dst)
                elif copy_mode == "hardlink":
                    try:
                        os.link(src, dst)
                    except Exception:
                        shutil.copy2(src, dst)
                elif copy_mode == "symlink":
                    try:
                        dst.symlink_to(src)
                    except Exception:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
                copied += 1

        triples += 1

    report = {
        "total_tifs": len(tifs),
        "total_jsons": len(jsons),
        "total_bases_found": len(by_base),
        "standardized_triples": triples,
        "incomplete_sets": incomplete_sets,
        "files_copied_or_linked": copied,
        "issues": issues,
        "standardized_dir": str(out_dir),
    }

    logger.info(f"Standardization report: {json.dumps(report, indent=2)}")
    
    if triples == 0:
        logger.warning("No complete triples found! This may indicate a data structure issue.")
        logger.info("Available files by type:")
        logger.info(f"  Images: {[p.name for p in images[:5]]}{'...' if len(images) > 5 else ''}")
        logger.info(f"  Labels: {[p.name for p in labels[:5]]}{'...' if len(labels) > 5 else ''}")
        logger.info(f"  JSONs: {[p.name for p in jsons[:5]]}{'...' if len(jsons) > 5 else ''}")
    
    return str(out_dir), report


def create_missing_label_files(data_dir: str, csv_path: str):
    """
    This function is not needed - we work with existing file structure.
    """
    logger.info("Using existing file structure - no label creation needed")
    return 0


def fix_data_structure(data_dir: str, csv_path: str):
    """
    Standardize into <data_dir>/_standardized and validate that triples exist.
    Returns (is_valid: bool, effective_data_dir: str)
    """
    std_dir, report = standardize_file_names(data_dir, csv_path)
    is_valid = report["standardized_triples"] > 0
    if not is_valid:
        logger.error("No complete {image,label,json} triples after standardization.")
    return is_valid, std_dir


def analyze_data_structure(data_dir: str):
    """
    Analyze the data directory structure and provide detailed diagnostics.
    This helps identify issues before running the experiment.
    """
    from pathlib import Path
    
    logger.info(f"Analyzing data structure in: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return None
    
    # Get all files (including subdirectories)
    tifs = list(data_path.rglob("*.tif"))
    jsons = list(data_path.rglob("*.json"))
    
    # Classify files
    images = []
    labels = []
    
    for p in tifs:
        name_lower = p.name.lower()
        if "label" in name_lower:
            labels.append(p)
        else:
            images.append(p)
    
    # Analyze naming patterns
    image_patterns = {}
    label_patterns = {}
    json_patterns = {}
    
    for img in images:
        pattern = _clean_base_id(img.name)
        image_patterns[pattern] = image_patterns.get(pattern, 0) + 1
    
    for lbl in labels:
        pattern = _clean_base_id(lbl.name)
        label_patterns[pattern] = label_patterns.get(pattern, 0) + 1
    
    for jsn in jsons:
        pattern = _clean_base_id(jsn.name)
        json_patterns[pattern] = json_patterns.get(pattern, 0) + 1
    
    # Find complete sets
    all_patterns = set(image_patterns.keys()) | set(label_patterns.keys()) | set(json_patterns.keys())
    complete_sets = []
    incomplete_sets = []
    
    for pattern in all_patterns:
        has_image = pattern in image_patterns
        has_label = pattern in label_patterns
        has_json = pattern in json_patterns
        
        if has_image and has_label and has_json:
            complete_sets.append(pattern)
        else:
            missing = []
            if not has_image: missing.append("image")
            if not has_label: missing.append("label")
            if not has_json: missing.append("json")
            incomplete_sets.append((pattern, missing))
    
    # Generate report
    analysis = {
        "total_files": len(tifs) + len(jsons),
        "total_images": len(images),
        "total_labels": len(labels),
        "total_jsons": len(jsons),
        "unique_patterns": len(all_patterns),
        "complete_sets": len(complete_sets),
        "incomplete_sets": len(incomplete_sets),
        "completion_rate": f"{len(complete_sets) / len(all_patterns) * 100:.1f}%" if all_patterns else "0%",
        "sample_image_names": [p.name for p in images[:5]],
        "sample_label_names": [p.name for p in labels[:5]],
        "sample_json_names": [p.name for p in jsons[:5]],
        "complete_patterns": complete_sets[:10],  # Show first 10
        "incomplete_examples": incomplete_sets[:10]  # Show first 10
    }
    
    if analysis['complete_sets'] == 0:
        logger.error("No complete data sets found! This will cause the experiment to fail.")
        for pattern, missing in analysis['incomplete_examples']:
            logger.warning(f"  Pattern '{pattern}': missing {', '.join(missing)}")
    
    return analysis


def organize_data_directory(data_dir: str):
    """
    Organize the data directory into a clean structure.
    
    Creates:
        - images/: Satellite imagery files
        - labels/: Ground truth label files  
        - metadata/: JSON metadata files
        - splits/: Data splits
        - experiments/: Experiment results
    """
    from pathlib import Path
    import shutil
    
    data_path = Path(data_dir)
    organized_dir = data_path / "organized"
    
    # Create organized directory first, then subdirectories
    organized_dir.mkdir(exist_ok=True)
    subdirs = ["images", "labels", "metadata", "splits", "experiments"]
    for subdir in subdirs:
        (organized_dir / subdir).mkdir(exist_ok=True)
    
    # Move files to appropriate directories
    moved_count = 0
    
    for file_path in data_path.iterdir():
        if file_path.is_file():
            # Skip the organized directory itself
            if "organized" in str(file_path):
                continue
                
            # Determine target directory based on file type
            if file_path.suffix == ".tif":
                if "label" in file_path.name.lower():
                    target_dir = organized_dir / "labels"
                else:
                    target_dir = organized_dir / "images"
            elif file_path.suffix == ".json":
                target_dir = organized_dir / "metadata"
            else:
                # Skip other file types
                continue
            
            # Move the file
            target_path = target_dir / file_path.name
            if not target_path.exists():
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
    
    # Move existing splits directory
    splits_dir = data_path / "splits"
    if splits_dir.exists():
        target_splits = organized_dir / "splits"
        if not target_splits.exists():
            shutil.move(str(splits_dir), str(target_splits))
    
    # Move _standardized directory if it exists
    std_dir = data_path / "_standardized"
    if std_dir.exists():
        # Move contents to organized structure
        for file_path in std_dir.rglob("*"):
            if file_path.is_file():
                if file_path.suffix == ".tif":
                    if "label" in file_path.name.lower():
                        target_dir = organized_dir / "labels"
                    else:
                        target_dir = organized_dir / "images"
                elif file_path.suffix == ".json":
                    target_dir = organized_dir / "metadata"
                else:
                    continue
                
                target_path = target_dir / file_path.name
                if not target_path.exists():
                    shutil.copy2(str(file_path), str(target_path))
                    moved_count += 1
        
        # Remove the _standardized directory
                shutil.rmtree(str(std_dir))
    
        return str(organized_dir)


def validate_data_structure(data_dir: str, csv_path: str = None):
    """
    Validate that data structure is correct for ML experiments.
    Enhanced version with better diagnostics.
    """
    from pathlib import Path

    # First analyze the structure
    analysis = analyze_data_structure(data_dir)
    if analysis is None:
        return {'valid': False, 'error': 'Data directory does not exist'}

    # Check if we have matching sets
    validation_result = {
        'valid': analysis['complete_sets'] > 0,
        'complete_sets': analysis['complete_sets'],
        'total_files': analysis['total_files'],
        'completion_rate': analysis['completion_rate'],
        'issues': []
    }

    if analysis['complete_sets'] == 0:
        validation_result['issues'].append("No complete data sets found")
        validation_result['issues'].append(f"Found {analysis['incomplete_sets']} incomplete sets")
    elif analysis['incomplete_sets'] > 0:
        validation_result['issues'].append(
            f"Found {analysis['incomplete_sets']} incomplete sets alongside {analysis['complete_sets']} complete ones"
        )

    # Check file count balance
    if analysis['total_images'] != analysis['total_labels']:
        validation_result['issues'].append(
            f"File count mismatch: {analysis['total_images']} images vs {analysis['total_labels']} labels"
        )

    if not validation_result['valid']:
        logger.error("Data structure validation failed")
        for issue in validation_result['issues']:
            logger.error(f"  - {issue}")

    return validation_result


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

    # ----- Proper file logging to run.log -----
    # Remove any existing FileHandlers to avoid duplicates
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    # ------------------------------------------

    try:
        logger.info(f"Starting experiment: {base_name}")

        # Resolve paths relative to workspace root
        data_dir = resolve_path(exp_cfg["data"]["data_dir"])
        csv_path = resolve_path(exp_cfg["data"]["csv_path"])

        # Check if data directory needs organization
        data_path = Path(data_dir)
        organized_dir = data_path / "organized"
        
        if not organized_dir.exists():
            organized_data_dir = organize_data_directory(data_dir)
            data_dir = organized_data_dir
            exp_cfg["data"]["data_dir"] = organized_data_dir
        else:
            data_dir = str(organized_dir)
            exp_cfg["data"]["data_dir"] = data_dir

        # First analyze the data structure to understand what we're working with
        analysis = analyze_data_structure(data_dir)
        if analysis is None:
            logger.error("Cannot proceed - data directory analysis failed")
            return

        # If we have complete sets in the organized directory, use it directly
        if analysis['complete_sets'] > 0:
            logger.info(f"Found {analysis['complete_sets']} complete data sets in organized directory")
            standardized_dir = data_dir
            structure_valid = True
        else:
            # Only run standardization if needed
            logger.info("Running standardization on organized directory")
            structure_valid, standardized_dir = fix_data_structure(
                data_dir=data_dir,
                csv_path=csv_path
            )

        # Use the appropriate directory for the rest of the pipeline
        exp_cfg["data"]["data_dir"] = standardized_dir

        if not structure_valid:
            # Re-analyze the standardized directory to see what we actually have
            std_analysis = analyze_data_structure(standardized_dir)
            if std_analysis and std_analysis['complete_sets'] == 0:
                logger.error("Even after standardization, no complete data sets found!")
                logger.error("The experiment will likely fail. Please check your data structure.")
                return

        # Check if cross-validation is enabled
        use_cross_validation = exp_cfg["data"].get("use_cross_validation", False)

        if use_cross_validation:
            return run_cv_experiment(exp_cfg, experiment_dir)
        else:
            return run_single_experiment(exp_cfg, experiment_dir)

    finally:
        logger.info(f"Logged output to {log_path}")


def run_single_experiment(exp_cfg, experiment_dir):
    """Run a single train/validation experiment."""
    use_auto_splitting = exp_cfg["data"].get("use_auto_splitting", True)

    # Track temporary directories for cleanup
    temp_dirs = []

    try:
        if use_auto_splitting:
            # Initialize data splitter with resolved paths
            splitter = IrrigationDataSplitter(
                csv_path=resolve_path(exp_cfg["data"]["csv_path"]),
                data_dir=resolve_path(exp_cfg["data"]["data_dir"]),
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

        # Prepare data with resolved paths
        data_dir = resolve_path(exp_cfg["data"]["data_dir"])
        label_bands = exp_cfg["data"]["label_bands"]

        logger.info(f"Creating datasets:")
        logger.info(f"  - Data directory: {data_dir}")
        logger.info(f"  - Train files: {len(train_files)}")
        logger.info(f"  - Val files: {len(val_files)}")
        logger.info(f"  - Label bands: {label_bands}")

        # Create datasets with file filtering
        train_dataset, train_temp_dir = create_filtered_dataset(data_dir, train_files, label_bands)
        val_dataset, val_temp_dir = create_filtered_dataset(data_dir, val_files, label_bands)

        # Track temp directories for cleanup
        if train_temp_dir:
            temp_dirs.append(train_temp_dir)
        if val_temp_dir:
            temp_dirs.append(val_temp_dir)

        logger.info(f"Dataset sizes:")
        logger.info(f"  - Train dataset: {len(train_dataset)} samples")
        logger.info(f"  - Val dataset: {len(val_dataset)} samples")

        if len(train_dataset) == 0:
            logger.error("Train dataset is empty! Cannot proceed with training.")
            return

        if len(val_dataset) == 0:
            logger.warning("Validation dataset is empty! Training will proceed without validation.")

        X_train, y_train = flatten_dataset(train_dataset)
        X_val, y_val = flatten_dataset(val_dataset)

        logger.info(f"Flattened data shapes:")
        logger.info(f"  - X_train: {X_train.shape}")
        logger.info(f"  - y_train: {y_train.shape}")
        if len(val_dataset) > 0:
            logger.info(f"  - X_val: {X_val.shape}")
            logger.info(f"  - y_val: {y_val.shape}")

        # Select only first two label bands for ML training/validation
        y_train = y_train[:, :2]
        if len(val_dataset) > 0:
            y_val = y_val[:, :2]

        # Train model
        model_type = exp_cfg["model"]["type"].lower()
        hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})

        logger.info(f"Training {model_type} model with hyperparameters: {hyperparams}")
        clf = train_model(X_train, y_train, model_type, **hyperparams)

        model_path = os.path.join(experiment_dir, "model.pkl")
        dump(clf, model_path)
        logger.info(f"Model saved to {model_path}")

        # Predict and evaluate if validation data exists
        if len(val_dataset) > 0:
            y_pred = clf.predict(X_val)
            metrics = model_metrics(y_pred, y_val)
            metrics_path = os.path.join(experiment_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

            num_samples = exp_cfg["visualization"].get("num_samples", 2)
            visualization_path = os.path.join(experiment_dir, "visualization.png")

            plot_ml_predictions(
                val_dataset, clf,
                num_samples=num_samples, save_path=visualization_path
            )
        else:
            logger.warning("No validation data available - skipping evaluation and visualization")

        # Optionally save feature importance
        save_feat_imp = exp_cfg.get("model", {}).get("save_feature_importance", False)
        if save_feat_imp and hasattr(clf, "estimators_"):
            try:
                BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDWI", "SCL"]
                N_TIMESTEPS = 37  # TODO: derive dynamically from dataset if needed
                featimp_path = os.path.join(experiment_dir, "feature_importance.csv")
                export_feature_importances(clf, BAND_NAMES, N_TIMESTEPS, featimp_path)
                # Generate and save band-by-time importance heatmap
                heatmap_path = os.path.join(experiment_dir, "band_time_importance.png")
                plot_band_time_importance(
                    featimp_path,
                    band_names=BAND_NAMES,
                    n_timesteps=N_TIMESTEPS,
                    save_path=heatmap_path
                )
            except Exception as e:
                logger.warning(f"Skipping feature-importance export due to error: {e}")
        logger.info(f"Experiment complete.")

    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def create_filtered_dataset(data_dir, file_list, label_bands):
    """
    Create a dataset that only includes the specified files.

    file_list entries can be:
      - full image filenames (e.g., 'ABC_20210301_image.tif')
      - bare ids (e.g., 'ABC_20210301'), in which case we look for '<id>_image.tif'
      - partial names that need pattern matching
    """
    from src.modeling.custom_dataset import MultiTemporalCropDataset
    import tempfile
    import shutil
    from pathlib import Path

    def _resolve_image_name(root: Path, item: str):
        """Enhanced file resolution with multiple fallback strategies"""
        # Strategy 1: Accept exact filename
        p = root / item
        if p.exists():
            return p
            
        # Strategy 2: Try common patterns with _image suffix
        candidates = [
            root / f"{item}_image.tif",
            root / f"{item}.tif",  # some datasets might already be named as the image
        ]
        for c in candidates:
            if c.exists() and "label" not in c.name.lower():
                return c
                
        # Strategy 3: Pattern matching - look for files starting with the item
        # This handles cases like 'site_-15.88_27.74_2021_53' -> 'site_-15.88_27.74_2021_53.tif'
        pattern_matches = list(root.rglob(f"{item}*.tif"))
        image_matches = [m for m in pattern_matches if "label" not in m.name.lower()]
        
        if image_matches:
            # Prefer exact matches first
            exact_matches = [m for m in image_matches if m.stem == item]
            if exact_matches:
                return exact_matches[0]
            # Then return the first match
            return image_matches[0]
            
        # Strategy 4: Look for files containing the item as a substring
        # This is more aggressive but handles edge cases
        for file_path in root.rglob("*.tif"):
            if "label" not in file_path.name.lower() and item in file_path.stem:
                return file_path
                
        return None

    def _find_companion_files(root: Path, image_path: Path):
        """Find label and JSON companion files for a given image"""
        base_name = image_path.stem
        
        # Try to find label file (search recursively)
        label_candidates = [
            root / f"{base_name}_label.tif",
            root / f"{base_name.replace('_image', '')}_label.tif",
            root / f"{base_name.replace('_image', '')}.tif"  # in case label doesn't have _label suffix
        ]
        
        label_path = None
        for candidate in label_candidates:
            if candidate.exists() and "label" in candidate.name.lower():
                label_path = candidate
                break
        
        # If not found in root, search recursively
        if label_path is None:
            for candidate in label_candidates:
                matches = list(root.rglob(candidate.name))
                for match in matches:
                    if "label" in match.name.lower():
                        label_path = match
                        break
                if label_path:
                    break
                
        # Try to find JSON file (search recursively)
        json_candidates = [
            root / f"{base_name}_image.json",
            root / f"{base_name.replace('_image', '')}_image.json",
            root / f"{base_name}.json"
        ]
        
        json_path = None
        for candidate in json_candidates:
            if candidate.exists():
                json_path = candidate
                break
        
        # If not found in root, search recursively
        if json_path is None:
            for candidate in json_candidates:
                matches = list(root.rglob(candidate.name))
                for match in matches:
                    json_path = match
                    break
                if json_path:
                    break
                
        return label_path, json_path

    temp_dir = tempfile.mkdtemp(prefix="filtered_data_")
    root = Path(data_dir)
    
    logger.info(f"Creating filtered dataset from {len(file_list)} file entries")
    logger.info(f"Source directory: {root}")
    logger.info(f"Temporary directory: {temp_dir}")

    successful_copies = 0
    failed_entries = []

    try:
        for entry in file_list:
            key = Path(entry).name  # tolerate paths
            logger.debug(f"Processing entry: {entry} -> key: {key}")
            
            img_path = _resolve_image_name(root, key)
            if not img_path:
                logger.warning(f"Could not find image file for '{entry}' in {root}")
                failed_entries.append(f"Image not found: {entry}")
                continue

            # Find companion files
            label_path, json_path = _find_companion_files(root, img_path)
            
            if not label_path:
                logger.warning(f"Could not find label file for image: {img_path.name}")
                failed_entries.append(f"Label not found for: {entry}")
                continue
                
            if not json_path:
                logger.warning(f"Could not find JSON file for image: {img_path.name}")
                failed_entries.append(f"JSON not found for: {entry}")
                continue

            # Copy all three files to temp directory
            files_to_copy = [
                (img_path, Path(temp_dir) / img_path.name),
                (label_path, Path(temp_dir) / label_path.name),
                (json_path, Path(temp_dir) / json_path.name)
            ]
            
            for src, dst in files_to_copy:
                try:
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied {src.name} to temp directory")
                except Exception as e:
                    logger.error(f"Failed to copy {src.name}: {e}")
                    failed_entries.append(f"Copy failed for {src.name}: {e}")
                    continue
            
            successful_copies += 1

        logger.info(f"Successfully processed {successful_copies} file sets")
        if failed_entries:
            logger.warning(f"Failed to process {len(failed_entries)} entries:")
            for failure in failed_entries[:5]:  # Show first 5 failures
                logger.warning(f"  - {failure}")
            if len(failed_entries) > 5:
                logger.warning(f"  ... and {len(failed_entries) - 5} more failures")

        if successful_copies == 0:
            logger.error("No files were successfully processed!")
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError("No valid file sets found")

        # Create dataset
        dataset = MultiTemporalCropDataset(
            image_dir=temp_dir,
            label_dir=temp_dir,
            label_bands=label_bands
        )
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset, temp_dir

    except Exception as e:
        logger.error(f"Error creating filtered dataset: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def run_cv_experiment(exp_cfg, experiment_dir):
    """Run cross-validation experiment."""
    # Track temporary directories for cleanup
    temp_dirs = []

    try:
        # Initialize data splitter with resolved paths
        splitter = IrrigationDataSplitter(
            csv_path=resolve_path(exp_cfg["data"]["csv_path"]),
            data_dir=exp_cfg["data"]["data_dir"],  # Use the standardized directory that was set earlier
            random_state=exp_cfg["data"].get("random_state", 42)
        )

        # Get CV parameters
        n_folds = exp_cfg["data"].get("n_folds", 5)
        cv_structure_name = exp_cfg["data"].get("cv_structure_name", "irrigation_cv")
        splits_dir = resolve_path(exp_cfg["data"].get("splits_dir", "./splits"))

        # Create CV folder structure
        cv_dir = splitter.create_cv_folder_structure(
            n_splits=exp_cfg["data"]["n_folds"],
            output_dir=exp_cfg["data"]["splits_dir"],
            structure_name=exp_cfg["data"]["cv_structure_name"],
            copy_files=exp_cfg["data"].get("copy_files", False),
            stratification_band=exp_cfg["data"]["stratification_band"]
        )
        logger.info(f"CV structure created at: {cv_dir}")

        # Prepare data processing parameters with resolved paths
        data_dir = resolve_path(exp_cfg["data"]["data_dir"])
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

            # Create datasets with file filtering
            train_dataset, train_temp_dir = create_filtered_dataset(data_dir, train_files, label_bands)
            val_dataset, val_temp_dir = create_filtered_dataset(data_dir, val_files, label_bands)

            # Track temp directories for cleanup
            if train_temp_dir:
                temp_dirs.append(train_temp_dir)
            if val_temp_dir:
                temp_dirs.append(val_temp_dir)

            if len(train_dataset) == 0:
                logger.error(f"Fold {fold_idx}: Train dataset is empty!")
                continue

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

            logger.info(f"Fold {fold_idx} metrics: {json.dumps(metrics, indent=2)}")

        # Aggregate results
        if fold_results:
            # Calculate mean and std of metrics across folds
            all_metrics = {}
            
            # Get the structure of metrics from the first fold
            metric_structure = fold_results[0]['metrics']
            
            # Aggregate each metric type separately
            for metric_type in metric_structure.keys():
                metric_means = {}
                metric_stds = {}
                
                # Get all values for this metric type across folds
                for metric_name in metric_structure[metric_type].keys():
                    values = [result['metrics'][metric_type][metric_name] for result in fold_results]
                    metric_means[metric_name] = float(np.mean(values))
                    metric_stds[metric_name] = float(np.std(values))
                
                all_metrics[f"{metric_type}_mean"] = metric_means
                all_metrics[f"{metric_type}_std"] = metric_stds

            # Add fold details
            all_metrics['n_folds_completed'] = len(fold_results)
            all_metrics['fold_details'] = fold_results

            # Save CV results
            cv_results_path = os.path.join(experiment_dir, "cv_results.json")
            with open(cv_results_path, "w") as f:
                json.dump(all_metrics, f, indent=2, default=str)

            logger.info(f"CV experiment complete. Results saved to {cv_results_path}")
            logger.info(f"Mean metrics across {len(fold_results)} folds:")
            for metric_type in metric_structure.keys():
                logger.info(f"  {metric_type}:")
                for metric_name in metric_structure[metric_type].keys():
                    mean_key = f"{metric_type}_mean"
                    if mean_key in all_metrics and metric_name in all_metrics[mean_key]:
                        logger.info(f"    {metric_name}: {all_metrics[mean_key][metric_name]:.4f}")
        else:
            logger.error("No folds completed successfully!")

    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def run_single_fold_experiment(exp_cfg, fold_number=1):
    """
    Run experiment on a single fold without cross-validation.
    This is useful for quick testing and debugging.
    """
    logger.info(f"Running single fold experiment on fold {fold_number}")
    
    # Resolve paths relative to workspace root
    data_dir = resolve_path(exp_cfg["data"]["data_dir"])
    csv_path = resolve_path(exp_cfg["data"]["csv_path"])
    
    logger.info(f"Resolved data_dir: {data_dir}")
    logger.info(f"Resolved csv_path: {csv_path}")
    
    # First analyze the data structure to understand what we're working with
    logger.info("=== Data Structure Analysis ===")
    analysis = analyze_data_structure(data_dir)
    if analysis is None:
        logger.error("Cannot proceed - data directory analysis failed")
        return
    
    # Standardize & validate
    logger.info("=== File Standardization ===")
    structure_valid, standardized_dir = fix_data_structure(
        data_dir=data_dir,
        csv_path=csv_path,
        label_bands=exp_cfg["data"]["label_bands"]
    )
    
    if not structure_valid:
        logger.error("Cannot proceed - data structure validation failed")
        return
    
    # Create data splitter
    splitter = IrrigationDataSplitter(
        data_dir=standardized_dir,
        csv_path=csv_path,
        random_state=exp_cfg["data"].get("random_state", 42)
    )
    
    # Create CV structure (but we'll only use one fold)
    cv_dir = splitter.create_cv_folder_structure(
        n_splits=exp_cfg["data"]["n_folds"],
        output_dir=exp_cfg["data"]["splits_dir"],
        structure_name=exp_cfg["data"]["cv_structure_name"],
        copy_files=exp_cfg["data"].get("copy_files", False),
        stratification_band=exp_cfg["data"]["stratification_band"]
    )
    
    # Get the specific fold we want to run
    fold_dir = os.path.join(cv_dir, "train", f"fold_{fold_number}")
    if not os.path.exists(fold_dir):
        logger.error(f"Fold {fold_number} directory not found: {fold_dir}")
        return
    
    # Get train and validation files for this fold
    train_files_path = os.path.join(fold_dir, "inner_train", "train_files.txt")
    val_files_path = os.path.join(fold_dir, "inner_val", "val_files.txt")
    
    if not os.path.exists(train_files_path) or not os.path.exists(val_files_path):
        logger.error(f"Train or validation files not found for fold {fold_number}")
        return
    
    # Read file lists
    with open(train_files_path, 'r') as f:
        train_files = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(val_files_path, 'r') as f:
        val_files = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Fold {fold_number}: {len(train_files)} train files, {len(val_files)} validation files")
    
    # Create datasets for this fold only
    train_dataset = create_filtered_dataset(standardized_dir, train_files, exp_cfg["data"]["label_bands"])
    val_dataset = create_filtered_dataset(standardized_dir, val_files, exp_cfg["data"]["label_bands"])
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Empty datasets created")
        return
    
    logger.info(f"Created train dataset with {len(train_dataset)} samples")
    logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    
    # Run the experiment on this single fold
    logger.info("=== Running Single Fold Experiment ===")
    
    # Build features
    X_train, y_train = flatten_dataset(train_dataset)
    X_val, y_val = flatten_dataset(val_dataset)
    
    logger.info(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Train model
    model = train_model(X_train, y_train, exp_cfg["model"])
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    metrics = model_metrics(y_val, y_pred, exp_cfg["data"]["label_bands"])
    
    logger.info(f"Single fold {fold_number} results:")
    for metric_type, metric_dict in metrics.items():
        for metric_name, value in metric_dict.items():
            logger.info(f"  {metric_type}_{metric_name}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run irrigation detection experiment")
    parser.add_argument("--single-fold", type=int, help="Run only a single fold (1-5) without CV")
    parser.add_argument("--config", type=str, default="experiment.yaml", help="Path to experiment config file")
    
    args = parser.parse_args()
    
    # Load experiment configuration
    exp_cfg = load_experiment(args.config)
    
    if args.single_fold:
        # Run single fold experiment
        if args.single_fold < 1 or args.single_fold > 5:
            print("Error: Fold number must be between 1 and 5")
            exit(1)
        
        print(f"Running single fold experiment on fold {args.single_fold}")
        run_single_fold_experiment(exp_cfg, args.single_fold)
    else:
        # Run full cross-validation experiment
        print("Running full cross-validation experiment")
        run_experiment(exp_cfg, args.config)