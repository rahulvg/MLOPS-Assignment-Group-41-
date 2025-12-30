import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import os
from train.data_loader import load_data
import joblib

DATA_PATH = "data/processed/heart_disease_clean.csv"


def run_experiments():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("HeartDisease-MLflow-MLOps")

    X, y = load_data(DATA_PATH)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    # ==================================================
    # Logistic Regression experiments
    # ==================================================
    print("\n===== Logistic Regression Experiments =====")

    lr_params = [0.1, 1.0, 10.0]

    for C in lr_params:
        run_name = f"LR_C{C}"

        with mlflow.start_run(run_name=run_name):
            print(f"\nRun: {run_name}")

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C,
                    solver="liblinear",
                    max_iter=1000
                ))
            ])

            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

            metrics = {}
            for m in scoring:
                mean = scores[f"test_{m}"].mean()
                std = scores[f"test_{m}"].std()
                metrics[f"{m}_mean"] = mean
                metrics[f"{m}_std"] = std
                print(f"{m.upper():10s} | Mean: {mean:.4f} | Std: {std:.4f}")
            
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("C", C)
            mlflow.log_metrics(metrics)

    # ==================================================
    # Random Forest experiments
    # ==================================================
    print("\n===== Random Forest Experiments =====")

    rf_params = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
    ]

    for params in rf_params:
        run_name = f"RF_ne{params['n_estimators']}_depth{params['max_depth']}"

        with mlflow.start_run(run_name=run_name):
            print(f"\nRun: {run_name}")

            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1
            )

            scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

            metrics = {}
            for m in scoring:
                mean = scores[f"test_{m}"].mean()
                std = scores[f"test_{m}"].std()
                metrics[f"{m}_mean"] = mean
                metrics[f"{m}_std"] = std
                print(f"{m.upper():10s} | Mean: {mean:.4f} | Std: {std:.4f}")

            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)




def train_and_save_final_model():
    """
    Train the final Logistic Regression (C=0.1) model and
    save it to both MLflow and local disk.
    """
    LOCAL_MODEL_DIR ="final_model"
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "heart_disease_lr_c01.pkl")
    # -------------------------------
    # MLflow setup
    # -------------------------------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("HeartDisease-MLflow-MLOps")

    print("Tracking URI:", mlflow.get_tracking_uri())

    # -------------------------------
    # Load data
    # -------------------------------
    X, y = load_data(DATA_PATH)

    # -------------------------------
    # Build final pipeline
    # -------------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            max_iter=1000
        ))
    ])

    # -------------------------------
    # Train & save
    # -------------------------------
    with mlflow.start_run(run_name="Final_LR_C0.1") as run:
        print("Run ID:", run.info.run_id)

        model.fit(X, y)

        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 0.1)
        mlflow.log_param("scaling", "StandardScaler")

        # Save to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="HeartDiseaseClassifier"
        )

        # Save locally
        joblib.dump(model, LOCAL_MODEL_PATH)

    print("\nFinal model saved successfully:")
    print("• MLflow registry: HeartDiseaseClassifier")
    print(f"• Local path: {LOCAL_MODEL_PATH}")

    return model


if __name__ == "__main__":
    run_experiments()


    # Logistic Regression with C=0.1 was selected as the final model because it achieved
    # the highest recall with low variance during cross-validated experiments.
    # In a medical risk prediction task, recall is prioritized to minimize false negatives
    # (i.e., missing patients with heart disease). The stronger regularization (lower C)
    # also improves generalization and model stability while retaining interpretability.


    train_and_save_final_model()
    