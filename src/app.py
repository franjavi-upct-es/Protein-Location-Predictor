from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import shutil
import os
import sys

# Importar el motor de inferencia
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference_engine import ProteinPredictor

# Inicializar la App y el Modelo
app = FastAPI(
    title="🧬 Protein Location Predictor API",
    description="Deep Learning Bio-Hybrid Model (ESM-2 + BioPhysic)",
    version="2.0",
)

# Variable global para el predictor
predictor = None


@app.on_event("startup")
def load_model():
    global predictor
    # Asumimos que el modelo está en /app/models dentro del contenedor
    model_path = os.getenv("MODEL_PATH", "models/esm2_hybrid_finetuned")
    print(f"Cargando modelo desde {model_path}...")
    try:
        # Usamos CPU para inferencia en servidor web estándar (más barato)
        predictor = ProteinPredictor(model_path, device="cpu")
        print("✅ Modelo cargado exitosamente.")
    except Exception as e:
        print(f"❌ Error fatal cargando el modelo: {e}")
        # No levantamos error aquí para permitir que la API arranque y reporte health check
        pass


class SequenceRequest(BaseModel):
    id: str = "prot_001"
    sequence: str


@app.get("/health")
def health_check():
    if predictor is None:
        return {"status": "loading_or_failed", "details": "Model not loaded yet"}
    return {"status": "healthy", "model_version": "ESM-2 Hybrid"}


@app.post("/predict")
def predict_single(request: SequenceRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="El modelo aún se está cargando.")

    # Validación básica
    clean_seq = request.sequence.upper().strip()
    if len(clean_seq) < 10:
        raise HTTPException(status_code=400, detail="La secuencia es demasiado corta.")

    try:
        result = predictor.predict(clean_seq)
        return {"id": request.id, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
