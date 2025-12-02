from xgboost import XGBClassifier


def build_model(random_state=42, **kwargs):
    """
    Construye y devuelve un modelo XGBClassifier.
    Este modelo es más potente que RandomForest para datasets de muchas características.
    """
    params = {"n_jobs": -1, "eval_metric": "mlogloss", "random_state": random_state}
    params.update(kwargs)

    model = XGBClassifier(**params)
    return model

