import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference_engine import ProteinPredictor


def explain_bio_features(model_dir, sample_sequences):
    """
    Genera un gráfico SHAP para entender cómo las propiedades físico-químicas
    afectan a la predicción del modelo.
    """
    print("🕵️‍♂️ Iniciando análisis de explicabilidad (SHAP)...")
    predictor = ProteinPredictor(model_dir, device="cpu")

    # 1. Preparar datos para el explainer
    # Necesitamos extraer solo la parte de la red que procesa bio-features
    # Wrapper para aislar la sub-red
    class BioNetWrapper(torch.nn.Module):
        def __init__(self, hybrid_model):
            super().__init__()
            self.projector = hybrid_model.bio_feature_projector
            self.classifier = hybrid_model.classifier
            # Necesitamos un vector 'dummy' de ESM para simular la concatenación
            # Usamos un vector de ceros o el promedio (simplificación para explicar solo bio)
            self.dummy_esm = torch.zeros(1, 1280)  # 1280 es dim de ESM-2 33

        def forward(self, x):
            # x son las bio_features normalizadas
            bio_emb = self.projector(x)
            # Simulamos que ESM no aporta nada para ver el peso PURO de la biofísica
            # Ojo: Esto es una aproximación
            combined = torch.cat((self.dummy_esm.repeat(x.shape[0], 1), bio_emb), dim=1)
            return self.classifier(combined)

    wrapper = BioNetWrapper(predictor.model)
    wrapper.eval()

    # 2. Generar datos de fondo (background) y de prueba
    print("Generando características...")
    data = []
    for seq in sample_sequences:
        bio = predictor.predict(seq)["bio_features"]
        vec = np.array([list(bio.values())])
        vec_norm = predictor.scaler.transform(vec)
        data.append(vec_norm[0])

    data_tensor = torch.tensor(np.array(data), dtype=torch.float32)

    # 3. Crear Explainer (DeepExplainer o KernelExplainer)
    # Usamos DeepExplainer que es eficiente para PyTorch
    explainer = shap.DeepExplainer(wrapper, data_tensor)

    shap_values = explainer.shap_values(data_tensor)

    # 4. Visualización
    feature_names = list(predictor.predict(sample_sequences[0])["bio_features"].keys())

    print("Generando gráfico de impacto...")
    # Resumen general
    shap.summary_plot(
        shap_values, data_tensor.numpy(), feature_names=feature_names, show=False
    )
    plt.savefig("reports/figures/shap_summary_biofeatures.png")
    print("✅ Gráfico guardado en reports/figures/shap_summary_biofeatures.png")


if __name__ == "__main__":
    # Secuencias de ejemplo
    sequences = [
        "MKILLDCLVMVLCG",  # Corta (ejemplo)
        "MDSKGSSQKGSRLLLLLVVSNLLLCQGVVS",  # Señal probable
        # ... añadir más secuencias reales para un buen análisis
    ] * 10

    explain_bio_features("models/esm2_hybrid_finetuned", sequences)

