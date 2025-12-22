import argparse
import sys
import os
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# Ajustar path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference_engine import ProteinPredictor

def process_fasta(fasta_path, output_path, model_dir):
    precitor = ProteinPredictor(model_dir)

    results = []

    # Leer FASTA con Biopython
    sequences = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"📂 Procesando {len(sequences)} secuencias de {fasta_path}...")

    for record in tqdm(sequences):
        seq = str(record.seq)
        if len(seq) > 10: # Ignorar fragmentos muy pequeños
            pred = precitor.predict(seq)

            row = {
                'id': record.id,
                'predicted_location': ", ".join(pred['predicted_locations']),
                'confidence_score': max(pred['probabilities'].values()),
                'is_anomalous': pred['is_ood'],
                'gravy_score': pred['bio_features']['gravy']
            }
            results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"✅ Resultados guardados en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicción masiva de localización de proteínas")
    parser.add_argument('--input', required=True, help="Archivo .fasta de entrada")
    parser.add_argument('--output', default="predictions.csv", help="Archivo .csv de salida")
    parser.add_argument('--model-dir', default='models/esm2_hybrid_finetuned', help="Directorio del modelo entrenado")

    args = parser.parse_args()
    process_fasta(args.input, args.output, args.model_dir)