import pandas as pd
import numpy as np
from collections import Counter
from itertools import product

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def calculate_kmer_composition(sequence: str, k: int) -> pd.Series:
    """
    Calcula la frecuencia de cada k-mer en una secuencia.
    Un kmer es una subsecuencia de longitud k
    """
    # 1. Generar todos los posibles k-mers
    # product(AMINO_ACIDS, repeat=k) genera ('A', 'A'), ('A', 'C'), ...
    all_kmers = ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]

    # Inicializar un diccionario de ceros para todos los k-mers
    composition = {kmer: 0 for kmer in all_kmers}

    # Validar la secuencia de entrada
    if not isinstance(sequence, str) or len(sequence) < k:
        return pd.Series(composition)

    # 3. Contar los k-mers en la secuencia
    total_kmers_in_sequence = len(sequence) - k + 1
    for i in range(total_kmers_in_sequence):
        kmer = sequence[i:i+k]
        if kmer in composition: # Solo contamos k-mers válidos
            composition[kmer] += 1

    # 4. Normalizar para obtener la frecuencia
    for kmer in composition:
        composition[kmer] /= total_kmers_in_sequence

    return pd.Series(composition)

def create_feature_matrix(input_path: str, output_path: str, k: int = 2):
    """
    Carga los datos procesados, crea la matriz de características (composición de AA)
    y la guarda junto con las etiquetas.
    """
    print(f"Cargando datos procesados de {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Creando matriz de características con k-mers (k={k})...")
    features_df = df['sequence'].apply(lambda seq: calculate_kmer_composition(seq, k=k))

    print(f"La nueva matriz de características tiene {features_df.shape[1]} dimensiones")

    # Combinamos las características con las etiquetas
    final_df = pd.concat([df[['accession', 'location_label']], features_df], axis=1)

    final_df.to_csv(output_path, index=False)
    print(f"Matriz de características guardada en {output_path}")
    return final_df

if __name__ == '__main__':
    create_feature_matrix(
        'data/processed/uniprot_processed_data.csv',
        'data/processed/features.csv',
        k=2
    )