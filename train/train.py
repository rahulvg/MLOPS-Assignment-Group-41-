# src/train.py

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data_loader import load_data


DATA_PATH = "data/processed/heart_disease_clean.csv"


def build_model():
    """
    Build preprocessing + model pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            max_iter=1000
        ))
    ])
    return pipeline


def train():
    """
    Train final model and log to MLflow.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("HeartDisease-MLflow-MLOps")

    X, y = load_data(DATA_PATH)
    model = build_model()

    with mlflow.start_run(run_name="CI_Training_Run") as run:
        model.fit(X, y)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 0.1)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HeartDiseaseClassifier"
        )

        print("Model trained and logged. Run ID:", run.info.run_id)


if __name__ == "__main__":
    train()
