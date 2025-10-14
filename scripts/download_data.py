import requests
import pandas as pd
from io import StringIO
import os
import re
import argparse

# --- CONSTANTES ---
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/stream"  # Usamos el endpoint 'stream' que es más eficiente
RAW_DATA_PATH = "data/raw/uniprot_data.csv"
PARAMS = {
    "query": "(organism_id:559292) AND (reviewed:true) AND (cc_subcellular_location:*)",
    "format": "tsv",
    "fields": "accession,sequence,cc_subcellular_location",
}


def get_next_link(headers):
    """
    Extrae la URL de la siguiente página de resultados de las cabeceras de la respuesta.
    """
    if "Link" in headers:
        match = re.search(r'<(.+?)>; rel="next"', headers["Link"])
        if match:
            return match.group(1)
    return None


def fetch_data_with_cursor(max_results: int = None):
    """
    Descarga datos de UniProt usando paginación por cursor, el método correcto para descargas masivas.
    """
    # Usamos requests.Session para mayor eficiencia
    session = requests.Session()
    # Hacemos la primera petición para obtener el primer lote y la URL del siguiente
    request = session.get(UNIPROT_API_URL, params=PARAMS, stream=True)
    request.raise_for_status()  # Lanza un error si la petición falla

    all_dfs = []
    total_results = 0

    # Bucle que se ejecuta mientras haya una página siguiente
    while True:
        # Leemos el contenido de la respuesta actual
        data_io = StringIO(request.text)
        df_page = pd.read_csv(data_io, sep='\t')
        all_dfs.append(df_page)

        # Actualizamos el contador y mostramos el progreso
        total_results += len(df_page)
        print(f"Descargados {len(df_page)} resultados... Total acumulado: {total_results}")

        # Comprobamos si hemos alcanzado el límite
        if max_results and total_results >= max_results:
            print("Límite de resultados alcanzado.")
            break

        # Buscamos el link de la siguiente página en las cabeceras
        next_link = get_next_link(request.headers)
        if next_link:
            # Si hay siguiente página, hacemos una nueva petición a esa URL
            request = session.get(next_link, stream=True)
            request.raise_for_status()
        else:
            # Si no hay 'next_link', hemos terminado
            print("No hay más páginas. Descarga completada.")
            break

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nDescarga total de {len(final_df)} filas antes del procesamiento.")

        # --- DE-DUPLICACIÓN Y LÍMITE ---
        duplicates_count = final_df['Entry'].duplicated().sum()
        if duplicates_count > 0:
            print(f"Se encontraron y eliminaron {duplicates_count} duplicados.")
            final_df.drop_duplicates(subset=['Entry'], keep='first', inplace=True)

        if max_results is not None and len(final_df) > max_results:
            final_df = final_df.head(max_results)
            print(f"DataFrame limitado a los primeros {len(final_df)} resultados únicos.")

        # --- RENOMBRADO Y GUARDADO ---
        final_df.rename(columns={
            'Entry': 'accession',
            'Sequence': 'sequence',
            'Subcellular location [CC]': 'subcellular_location'
        }, inplace=True)

        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        final_df.to_csv(RAW_DATA_PATH, index=False)
        print(f"\nProceso finalizado. Se han guardado {len(final_df)} proteínas únicas en {RAW_DATA_PATH}")
    else:
        print("No se han descargado datos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga datos de UniProt.")
    parser.add_argument(
        "--max_results",
        type=int,
        default=500,
        help="Número máximo de resultados a descargar. Utiliza -1 para que se descarguen todas las muestras"
    )
    args = parser.parse_args()

    # Convertir -1 en None para la función
    max_results_to_fetch = None if args.max_results == -1 else args.max_results

    fetch_data_with_cursor(max_results=max_results_to_fetch)