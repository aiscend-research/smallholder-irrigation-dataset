from ml_pipeline.build_features import *
from ml_pipeline.ml_model import *
from ml_pipeline.evaluation import *
from ml_pipeline.visualization import *


def main():
    dataset_path = "multi-temporal-crop-classification-subset"

    # 1. Get datamodule & setup train/test datasets
    datamodule = get_datamodule(dataset_path)
    datamodule.setup("fit")
    train_dataset = datamodule.train_dataset
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset

    # 2. Flatten data
    X_train, y_train = flatten_dataset(train_dataset)
    X_test, y_test = flatten_dataset(test_dataset)

    # 3. Set up class names and colors
    class_names = train_dataset.class_names
    colors = [
        "#0000FF", "#000088", "#00FF00", "#00FFFF", "#006600",
        "#A9A9A9", "#3399FF", "#FFFF00", "#FFCC00",
        "#FFA500", "#FF0000", "#990000", "#800000"
    ]

    # 4. Train model on small subset
    clf = train_randomForest(X_train[:1000], y_train[:1000])

    # 5. Predict and evaluate
    y_pred = clf.predict(X_test[:1000])
    print(model_metrics(y_pred, y_test[:1000])) #accuracy & f1 score

    # 6. Visualize results
    print_confusion_matix(y_test[:1000], y_pred)
    plot_ml_predictions(test_dataset, clf, class_names, colors, num_samples=2)

if __name__ == "__main__":
    main()