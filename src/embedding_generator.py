import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel
import os
import sys

# Importar la función de feature engineering
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import calculate_kmer_composition

# --- CONFIGURACIÓN ---
DATA_PATH = "data/processed/uniprot_processed_data.csv"
OUTPUT_PATH = "data/processed/embeddings.csv"
OUTPUT_COMBINED_PATH = "data/processed/embeddings_with_kmers.csv"
# Usaremos un modelo pequeño de ESM-2 para que sea manejable en un PC estándar
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
KMER_SIZE = 3  # Tamaño de k-mers a usar


def generate_embeddings():
    """
    Carga secuencias, genera embeddings usando un modelo ESM-2 y los guarda.
    """
    print("Cargando datos procesados...")
    df = pd.read_csv(DATA_PATH)
    sequences = df["sequence"].tolist()

    print(f"Cargando el modelo pre-entrenado ESM-2: {MODEL_NAME}...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)

    # Si tienes una GPU compatible con CUDA o Metal (Apple Silicon), la usaremos para acelerar el proceso
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model = model.to(device)

    print(f"Usando el dispositivo: {device}")

    embeddings = []
    print(f"Generando embeddings para {len(sequences)} secuencias...")

    # Procesamos las secuencias en lotes para no saturar la memoria
    batch_size = 16
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i : i + batch_size]

        # Tokenizar: convierte la secuencia de texto en IDs que el modelo entiende
        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1022,
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Desactivamos el cálculo de gradientes para ahorrar memoria
        with torch.no_grad():
            # Pasamos los datos al modelo
            outputs = model(**inputs)

        # Obtenemos los embeddings y calculamos una representación de toda la secuencia
        # promediando los embeddings de todos sus aminoácidos.
        for seq_embedding in outputs.last_hidden_state:
            # Movemos el tensor de vuelta a la CPU para los cálculos con numpy
            mean_embedding = seq_embedding.mean(dim=0).cpu().numpy()
            embeddings.append(mean_embedding)

        print(
            f"Procesado lote {i // batch_size + 1}/{(len(sequences) - 1) // batch_size + 1}"
        )

    print("Generación de embeddings completada.")

    # Creamos un DataFrame con los embeddings
    embedding_df = pd.DataFrame(embeddings)

    # Renombrar columnas de embeddings para mayor claridad
    embedding_df.columns = [f"emb_{i}" for i in range(embedding_df.shape[1])]

    # Añadimos las etiquetas y el accession para referencia
    final_df = pd.concat([df[["accession", "location_label"]], embedding_df], axis=1)

    # Guardamos los embeddings
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Embeddings guardados en {OUTPUT_PATH}")

    # --- GENERAR K-MERS Y COMBINAR ---
    print(f"\nGenerando características de k-mers (k={KMER_SIZE})...")
    kmer_features = df["sequence"].apply(
        lambda seq: calculate_kmer_composition(seq, k=KMER_SIZE)
    )

    # Renombrar columnas de k-mers para mayor claridad
    kmer_features.columns = [f"kmer_{col}" for col in kmer_features.columns]

    print(f"K-mers generados: {kmer_features.shape[1]} características")

    # Combinar embeddings y k-mers
    combined_df = pd.concat(
        [df[["accession", "location_label"]], embedding_df, kmer_features], axis=1
    )

    # Guardar el dataset combinado
    combined_df.to_csv(OUTPUT_COMBINED_PATH, index=False)
    print(f"Dataset combinado (embeddings + k-mers) guardado en {OUTPUT_COMBINED_PATH}")
    print(
        f"Total de características: {embedding_df.shape[1]} embeddings + {kmer_features.shape[1]} k-mers = {embedding_df.shape[1] + kmer_features.shape[1]}"
    )


if __name__ == "__main__":
    generate_embeddings()

