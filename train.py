"""Train and save an Iris species classifier with a held-out accuracy report."""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iris_classifier.joblib"


def build_model() -> object:
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )


def main() -> None:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    evaluation_model = build_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Held-out accuracy (20% stratified test split): {accuracy:.3f}")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # Refit on all available examples for the model used by the API.
    production_model = build_model()
    production_model.fit(iris.data, iris.target)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": production_model,
            "target_names": iris.target_names.tolist(),
            "features": iris.feature_names,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
