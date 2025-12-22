import os
import torch
import gc
from transformers import EsmTokenizer, EsmConfig, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import joblib
import pandas as pd

# Importar nuestras clases personalizadas
from dataset import HybridProteinDataset
from hybrid_model import ESM2WithBioFeatures

# --- CONFIGURACIÓN ---
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = "data/processed/uniprot_processed_data.csv"
OUTPUT_DIR = "models/esm2_hybrid_finetuned"
MAX_LENGTH = 1022
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8

def cleanup():
    """Limpia la caché de la GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Sigmoide para probabilidad
    probs = 1 / (1 + np.exp(-predictions))
    y_pred = (probs > 0.5).astype(int)

    return {
        "f1_macro": f1_score(labels, y_pred, average="macro"),
        "accuracy": accuracy_score(labels, y_pred),
    }


def main():
    print("🧬 Iniciando entrenamiento híbrido (ESM-2 + BioFísica)...")

    # Configuración de memoria PyTorch (Ayuda a la fragmentación)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)

    # 1. Carga y División de datos
    full_df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    train_df.to_csv('data_train_tmp.csv', index=False)
    val_df.to_csv('data_val_tmp.csv', index=False)

    # 2. Instanciar Datasets
    print("Creando datasets y calculando características")
    # Dataset de entrenamiento: Ajusta el Scaler y el Encoder
    train_dataset = HybridProteinDataset('data_train_tmp.csv', tokenizer, max_length=MAX_LENGTH)

    # Dataset de validación: Usa el Scaler y Encoder ya ajustados (para evitar data leakage)
    val_dataset = HybridProteinDataset(
        'data_val_tmp.csv',
        tokenizer,
        max_length=MAX_LENGTH,
        label_encoder=train_dataset.mlb,
        scaler=train_dataset.scaler
    )

    num_extra_features = train_dataset.bio_features.shape[1]
    num_labels = len(train_dataset.mlb.classes_)
    print(f"--> Características extra: {num_extra_features}")
    print(f"--> Clases a predecir: {num_labels}")

    # 3. Inicializar Modelo Híbrido
    config = EsmConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = ESM2WithBioFeatures.from_pretrained(MODEL_NAME, config=config, num_extra_features=num_extra_features)

    # Congelamos parte de ESM-2 para acelerar y enfocar el aprendizaje en la fusión
    print("❄️ Congelando capas base de ESM-2 para ahorrar memoria...")
    for param in model.esm.embeddings.parameters():
        param.requires_grad = False

    # ESM-2 t33 tiene 33 capas. Congelamos las primeras 28.
    for layer in model.esm.encoder.layer[:28]:
        for param in layer.parameters():
            param.requires_grad = False

    # 4. Configurar Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4, # LR muy bajo para fine-tuning
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        logging_dir=f"{OUTPUT_DIR}/logs",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False, # IMPORTANTE: Para que no borre 'extra_features' del dataset'
        dataloader_num_workers=1,
        optim='adamw_torch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Guardar todo
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_hybrid_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_hybrid_model"))
    joblib.dump(train_dataset.scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(train_dataset.mlb, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    cleanup()

    print("✅ Entrenamiento híbrido completado.")

    # Limpieza
    os.remove('data_train_tmp.csv')
    os.remove('data_val_tmp.csv')

if __name__ == "__main__":
    import numpy as np
    main()