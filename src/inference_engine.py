import torch
import numpy as np
import pandas as pd
import joblib
import os
from transformers import EsmTokenizer, EsmConfig
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Importamos nuestras clases
from hybrid_model import ESM2WithBioFeatures
from bio_features import calculate_physicochemical_props

class ProteinPredictor:
    def __init__(self, model_dir, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ Cargando motor de inferencia en {self.device}...")

        # 1. Cargar Artefactos
        self.tokenizer = EsmTokenizer.from_pretrained(os.path.join(model_dir, "final_hybrid_model"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.mlb = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        
        # 2. Reconstruir Modelo Híbrido
        # Necesitamos saber num_extra_features para inicializar la arquitectura
        # Lo inferimos del scaler (mean_ tiene el tamaño de features)
        num_extra_features = self.scaler.mean_.shape[0]
        num_labels = len(self.mlb.classes_)
        
        config = EsmConfig.from_pretrained(os.path.join(model_dir, "final_hybrid_model"), num_labels=num_labels)
        self.model = ESM2WithBioFeatures.from_pretrained(
            os.path.join(model_dir, "final_hybrid_model"), 
            config=config,
            num_extra_features=num_extra_features
        )
        self.model.to(self.device)
        self.model.eval()
        print("✅ Modelo cargado y listo.")

    def _calculate_entropy(self, probs):
        """Calcula la entropía de Shanon para detectar incertidumbre"""
        return -np.sum(probs * np.log(probs + 1e-9))

    def predict(self, sequence, threshold=0.5, ood_threshold=1.5):
        """
        Predice la localización.
        - threshold: umbral para considerar una clase activa (Multi-Label).
        - ood_threshold: umbral de entropía para marcar como 'Desconocido'.
        """
        # A. Preprocesamiento Biofísico
        bio_props = calculate_physicochemical_props(sequence)
        bio_vector = np.array([list(bio_props.values())])
        bio_vector_norm = self.scaler.transform(bio_vector)
        bio_tensor = torch.tensor(bio_vector_norm, dtype=torch.float32).to(self.device)

        # B. Preprocesamiento Secuencia
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1022
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # C. Inferencia
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                extra_features=bio_tensor
            )
            logits = outputs['logits']
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # D. Lógica OOD y Post-procesamiento
        entropy = self._calculate_entropy(probs)
        
        # Obtenemos etiquetas que superan el umbral
        active_indices = np.where(probs > threshold)[0]
        predicted_labels = [self.mlb.classes_[i] for i in active_indices]
        
        results = {
            "predicted_locations": predicted_labels,
            "probabilities": {cls: float(p) for cls, p in zip(self.mlb.classes_, probs)},
            "is_ood": False,
            "entropy": float(entropy),
            "bio_features": bio_props
        }

        # Si la entropía es muy alta o ninguna probabilidad supera el umbral
        if len(predicted_labels) == 0 or entropy > ood_threshold:
            results["is_ood"] = True
            if len(predicted_labels) == 0:
                results["predicted_locations"] = ["Unknown/Low Confidence"]

        return results