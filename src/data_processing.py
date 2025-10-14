import pandas as pd
import argparse

def clean_location(location_string: str) -> str:
    """Extrae la localización principal de la cadena de texto de UniProt"""
    if isinstance(location_string, str):
        # Limpia el prefijo y toma la primera localización mencionada antes de detalles o evidencias
        main_location = location_string.replace("SUBCELLULAR LOCATION: ", "").split(".")[0].split(" {")[0].strip()
        return main_location
    return None

def group_locations_hierarchical(label: str) -> str:
    """
    Agrupa etiquetas de localización detalladas en categorías más generales
    usando una lógica de prioridad para evitar ambigüedades
    """
    if not isinstance(label, str):
        return 'Unknown'

    # Convertir a minúsculas para una comparación robusta
    label_lower = label.lower()

    # El orden de estas comprobaciones es CRUCIAL
    if 'nucleus' in label_lower or 'nucleolus' in label_lower:
        return 'Nucleus'
    if 'mitochondrion' in label_lower or 'mitochondria' in label_lower:
        return 'Mitochondrion'
    if 'endoplasmic reticulum' in label_lower:
        return 'Endoplasmic Reticulum'
    if 'golgi' in label_lower:
        return 'Golgi Apparatus'
    if 'peroxisome' in label_lower:
        return 'Peroxisome'
    if 'vacuole' in label_lower:
        return 'Vacuole'
    if 'secreted' in label_lower or 'cell wall' in label_lower:
        return 'Secreted/Extracellular'
    if 'cytoplasm' in label_lower or 'cytosol' in label_lower or 'cytoskeleton' in label_lower:
        return 'Cytoplasm'
    # La comprobación de 'Membrane' va al final para capturar membranas
    # no asociadas a un orgánulo específico ya clasificado.
    if 'membrane' in label_lower:
        return 'Membrane'
    if 'lipid droplet' in label_lower:
        return 'Lipid Droplet'

    # Si no encaja en ninguna categoría principal, la marcamos para posible descarte.
    return 'Other'

def process_data(input_path: str, output_path: str, min_samples: int):
    """
    Carga los datos crudos, los limpia, agrupa las clases de forma jerárquica
    y guarda los datos procesados.
    """
    print(f"Cargando datos crudos de {input_path}...")
    df_raw = pd.read_csv(input_path)
    df_processed = df_raw.copy()

    # 1. Limpieza inicial
    df_processed['location_label'] = df_processed['subcellular_location'].apply(clean_location)
    df_processed.dropna(subset=['location_label', 'sequence'], inplace=True)

    # 2. Aplicamos la nueva lógica de agrupación jerárquica
    print("Agrupando etiquetas de localización de forma jerárquica...")
    df_processed['location_label'] = df_processed['location_label'].apply(group_locations_hierarchical)

    # 3. Excluimos las clases 'Other' y 'Unknown'
    df_processed = df_processed[~df_processed['location_label'].isin(['Other', 'Unknown'])]

    # 4. Filtramos para quedarnos con clases suficientemente representadas
    # Este umbral es importante para tener un modelo robusto
    min_samples_per_class = min_samples
    value_counts = df_processed['location_label'].value_counts()
    print("\nConteo de clases después de agrupar:")
    print(value_counts)

    top_locations = value_counts[value_counts >= min_samples_per_class].index.tolist()
    df_final = df_processed[df_processed['location_label'].isin(top_locations)]

    print(f"Clases finales a utilizar (con >= {min_samples_per_class} muestras): {top_locations}")

    # 5. Guardar los datos procesados
    df_final.to_csv(output_path, index=False)
    print(f"Datos procesados y guardados en {output_path}. Total de filas: {len(df_final)}")
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa datos de UniProt.")
    parser.add_argument(
        "--min_samples",
        type=int,
        default=50,
        help="Umbral mínimo requerido para que una clase se incluya en el conjunto de datos."
    )
    args = parser.parse_args()

    # Esto permite ejecutar el script directamente para probarlo
    process_data(
        "data/raw/uniprot_data.csv",
        "data/processed/uniprot_processed_data.csv",
        min_samples=args.min_samples
    )