import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import optuna

from model import build_model

# Definimos las rutas a los archivos
# Características combinadas
FEATURES_PATH = "data/processed/embeddings_with_kmers.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "protein_location_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


def objective(trial, X, y):
    """
    Función objetivo para Optuna. Entrena y evalúa un modelo con params sugeridos.
    """
    # Espacio de búsqueda de hiperparámetros
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        'tree_method': 'hist',
        'device': 'cuda',
    }

    # Validación cruzada manual para manejar sample_weights correctamente
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Recalcular pesos para este fold
        classes_fold = np.unique(y_train_fold)
        weights = compute_class_weight("balanced", classes=classes_fold, y=y_train_fold)
        sample_weights_fold = np.array([weights[label] for label in y_train_fold])

        # Entrenar modelo temporal
        model = build_model(**param)
        model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)

        preds = model.predict(X_val_fold)
        # Usamos F1 Weighted porque nos importa el balance entre clases
        scores.append(f1_score(y_val_fold, preds, average="weighted"))

    return np.mean(scores)


def tune_hyperparameters(X, y):
    """Ejecuta el estudio de Optuna"""
    print("--- Iniciando optimización de hiperparámetros con Optuna ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=30)  # 30 intentos

    print(f"Mejores parámetros encontrados: {study.best_params}")
    return study.best_params


def train():
    print("Cargando características combinadas...")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"No se encontró: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["accession", "location_label"])
    y = df["location_label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- FASE DE OPTIMIZACIÓN ---
    # Nota: Si esto tarda mucho, se reducen n_trial en la función tune_hyperparameters
    best_params = tune_hyperparameters(X_train, y_train)

    # --- ENTRENAMIENTO FINAL ---
    print("\nEntrenando el modelo final con los mejores hiperparámetros")
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    sample_weights = np.array([class_weights[label] for label in y_train])

    # Pasamos los mejores parámetros al constructor
    model = build_model(**best_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )

    print(f"\n--- Resultados Finales ---")
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Reporte:\n{report}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    train()
