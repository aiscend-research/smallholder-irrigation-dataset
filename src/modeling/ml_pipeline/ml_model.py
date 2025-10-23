from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler

def train_random_forest(
    X_train, y_train,
    n_estimators=100,
    random_state=42,
    use_smote=True,
    sampling_strategy='auto',
    k_neighbors=5
):
    """
    Train a Random Forest model with optional SMOTE oversampling
    for imbalanced classification tasks.
    """
    # Base model
    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None  # handled by SMOTE balance instead
    )

    # Choose oversampling method
    if use_smote:
        resampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    else:
        resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

    # Handle multi-output case
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        clf = Pipeline([
            ("smote", resampler),
            ("clf", base_model)
        ])
    else:
        # MultiOutputClassifier doesn’t directly support pipelines inside,
        # so we wrap base model only
        clf = MultiOutputClassifier(base_model)

    clf.fit(X_train, y_train)
    return clf


def train_GradientBoosting(
    X_train, y_train,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    random_state=42
):
    """
    Train a Gradient Boosting model (no built-in SMOTE since GB is slow).
    """
    base_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )

    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        clf = base_model
    else:
        clf = MultiOutputClassifier(base_model)

    clf.fit(X_train, y_train)
    return clf


def train_model(X_train, y_train, model_type, **hyperparams):
    if model_type == "random_forest":
        clf = train_random_forest(X_train, y_train, **hyperparams)
    elif model_type == "gradient_boosting":
        clf = train_GradientBoosting(X_train, y_train, **hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return clf
