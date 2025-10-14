import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

from model import build_model

# Definimos las rutas a los archivos
FEATURES_PATH = 'data/processed/embeddings.csv'
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'protein_location_model.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

def train():
    """
    Función principal para entrenar, evaluar y guardar el modelo.
    """
    # 1. Cargar datos
    print("Cargando matriz de características")
    df = pd.read_csv(FEATURES_PATH)

    # 2. Preparar datos
    X = df.drop(columns=['accession', 'location_label'])
    y = df['location_label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    classes = np.unique(y_train)

    print(f"Datos divididos: {len(X_train)} para entrenamiento, {len(X_test)} para prueba.")

    print("Calculando pesos de muestra para manejar el desbalance de clases...")
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[label] for label in y_train])

    # 3. Construir y entrenar el modelo
    print("Construyendo y entrenando el modelo XGBoost...")
    model = build_model()
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 4. Evaluar el modelo
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print(f"\n--- Resultados de la Evaluación ---")
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print(f"Reporte de Clasificación:\n{report}")
    print("-" * 80)

    # 5. Guardar el modelo y el codificador
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Codificador guardado en: {ENCODER_PATH}")

if __name__ == "__main__":
    train()