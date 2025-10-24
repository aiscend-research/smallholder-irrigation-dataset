from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

def _filter_params(estimator_cls, params: dict) -> dict:
    """Keep only kwargs that the estimator actually supports."""
    valid = estimator_cls().get_params().keys()
    return {k: v for k, v in params.items() if k in valid}

def train_random_forest(
    X_train, y_train,
    n_estimators=100,
    random_state=42,
    **kwargs
):
    # Defaults; allow override via kwargs (e.g., class_weight)
    defaults = dict(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    # Merge and filter to what RF supports
    rf_params = {**defaults, **kwargs}
    rf_params = _filter_params(RandomForestClassifier, rf_params)

    base_model = RandomForestClassifier(**rf_params)

    # Single-label vs multi-output
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        clf = base_model
    else:
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
    random_state=42,
    **kwargs
):
    defaults = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )
    gb_params = {**defaults, **kwargs}
    gb_params = _filter_params(GradientBoostingClassifier, gb_params)  # drops class_weight, etc.

    base_model = GradientBoostingClassifier(**gb_params)

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