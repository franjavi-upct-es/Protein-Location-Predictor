import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np


def calculate_physicochemical_props(sequence: str):
    """
    Calcula propiedades físico-químicas usando Biopython.
    Retorna un diccionario con las características.
    """
    # Limpieza básica: Eliminar caracteres no estándar (X, B, Z, J, O, U)
    # Reemplazamos por G (Glicina) o A (Alanina) para no afectar drásticamente el peso/carga
    clean_seq = (
        sequence.replace("X", "A")
        .replace("B", "N")
        .replace("Z", "Q")
        .replace("U", "C")
        .replace("O", "K")
        .replace("J", "L")
    )

    try:
        analyser = ProteinAnalysis(clean_seq)

        props = {}

        # 1. Peso Molecular: Proteínas nucleares suelen ser grandes complejos, secretadas más pequeñas
        props["molecular_weight"] = analyser.molecular_weight()

        # 2. Aromaticidad: Relacionada con estabilidad y entornos hidrofóbicos
        props["aromaticity"] = analyser.aromaticity()

        # 3. Inestabilidad: Índice que predice si la proteína es estable en tubo de ensayo
        props["instability_index"] = analyser.instability_index()

        # 4. Punto Isoeléctrico (pI): CRUCIAL.
        # Proteínas nucleares/mitocondriales suelen ser básicas (pI > 7).
        # Proteínas citosólicas suelen ser ligeramente ácidas.
        props["isoelectric_point"] = analyser.isoelectric_point()

        # 5. Fracción de Estructura Secundaria (Hélice, Hoja, Giro)
        # Importante para proteínas de membrana (ricas en hélices)
        helix, turn, sheet = analyser.secondary_structure_fraction()
        props["helix_fraction"] = helix
        props["turn_fraction"] = turn
        props["sheet_fraction"] = sheet

        # 6. FRAVY (Grand Average of Hydripathy)
        # Valor positivo = Hidrofóbico (Membrana)
        # Valor negativo = Hidrofílico (Citosol/Núcleo)
        props["gravy"] = analyser.gravy()

        return props

    except Exception as e:
        # En caso de eror (seq muy corta o inválido), devolvemos ceros
        return {
            "molecular_weight": 0,
            "aromaticity": 0,
            "instability_index": 0,
            "isoelectric_point": 7,
            "helix_fraction": 0,
            "turn_fraction": 0,
            "sheet_fraction": 0,
            "gravy": 0,
        }


def add_bio_features(input_path, output_path):
    print(f"Calculando propiedades físico-químicas para {input_path}...")
    df = pd.read_csv(input_path)

    # Aplicar cálculo
    # Esto devuelve una serie de diccionarios, que convertimos a DataFrame
    bio_features = (
        df["sequence"].apply(calculate_physicochemical_props).apply(pd.Series)
    )

    # Prefijo para identificar estas columnas
    bio_features.columns = [f"bio_{col}" for col in bio_features.columns]

    print(f"Generadas {bio_features.shape[1]} nuevas características biológicas.")

    # Concatenar con el dataframe original
    # Asumimos que el input_path es el archivo que ya tiene embeddings o datos procesados
    df_final = pd.concat([df, bio_features], axis=1)

    df_final.to_csv(output_path, index=False)
    print(f"Datos enriquecidos guardados en {output_path}")


if __name__ == "__main__":
    # Ejemplo de uso
    add_bio_features(
        "data/processed/uniprot_processed_data.csv",
        "data/processed/bio_features_data.csv",
    )
