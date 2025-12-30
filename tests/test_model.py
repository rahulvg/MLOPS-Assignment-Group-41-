from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from train.train_experiment import train_and_save_final_model


def test_model_pipeline_structure():
    """Final model should be a pipeline with scaler and classifier."""
    model = train_and_save_final_model()

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "clf" in model.named_steps

    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["clf"], LogisticRegression)


def test_model_hyperparameters():
    """Verify final model hyperparameters."""
    model = train_and_save_final_model()
    clf = model.named_steps["clf"]

    assert clf.C == 0.1
    assert clf.solver == "liblinear"


def test_model_can_predict():
    """Trained model should be able to make predictions."""
    model = train_and_save_final_model()

    # Dummy input with correct number of features
    n_features = model.named_steps["clf"].coef_.shape[1]
    import numpy as np
    X_dummy = np.zeros((1, n_features))

    preds = model.predict(X_dummy)

    assert preds.shape == (1,)
