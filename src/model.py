from xgboost import XGBClassifier

def build_model(random_state=42):
    """
    Construye y devuelve un modelo XGBClassifier.
    Este modelo es más potente que RandomForest para datasets de muchas características.
    """
    model = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    return model