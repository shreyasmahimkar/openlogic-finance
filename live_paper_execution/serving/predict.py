"""Model serving (Box 5) — the *deploy* surface for the return/regime model.

Loads a persisted model and returns calibrated probabilities + regimes. This is
the local serving function; in production the same `ReturnRegimeModel` is hosted
on a **Vertex AI** or **AWS SageMaker** endpoint (cross-cloud), promoted only
after the validation gate passes (see strategy_testing/validation).
"""

from model_library.ml_zoo.return_regime import ReturnRegimeModel


def load_and_predict(model_path: str, X):
    """Load a saved model and score a feature frame.

    Returns:
        (probabilities_up, regimes) — P(up) in [0,1] and the mapped regime labels.
    """
    model = ReturnRegimeModel.load(model_path)
    return model.predict_proba_up(X), model.predict_regime(X)
