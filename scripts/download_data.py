import requests
import pandas as pd
from io import StringIO
import os
import re
import argparse

# --- CONSTANTES ---
UNIPROT_API_URL = "https://rest.uniprot.org/uniprotkb/stream"
RAW_DATA_PATH = "data/raw/uniprot_data_multispecies.csv"

# IDs de Taxonomía para organismos modelo:
# 9606: Homo sapiens (Humano)
# 10090: Mus musculus (Ratón)
# 559292: Saccharomyces cerevisiae (Levadura )
# 83333: Escherichia coli (Bacteria modelo)
# 3702: Arabidopsis thaliana (Planta modelo)
DEFAULT_TAXON_IDS = [9606, 559292, 83333, 10090, 3702]


def get_next_link(headers):
    if "Link" in headers:
        match = re.search(r'<(.+?)>; rel="next"', headers["Link"])
        if match:
            return match.group(1)
    return None


def build_query(taxon_ids):
    """Construye una query OR para múltiples organismos"""
    taxons_str = " OR ".join([f"organism_id:{tid}" for tid in taxon_ids])
    # Filtramos por revisado (Swiss-Prot) y que tenga localización conocida
    return f"({taxons_str}) AND (reviewed:true) AND (cc_subcellular_location:*)"


def fetch_data_with_cursor(
    max_results: int = None, taxon_ids: list = DEFAULT_TAXON_IDS
):
    session = requests.Session()

    query = build_query(taxon_ids)
    print(f"🧬 Query construida para {len(taxon_ids)} especies: {query}")

    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,sequence,cc_subcellular_location,organism_name",  # Añadimos organism_name para control
        "size": 500,
    }

    request = session.get(UNIPROT_API_URL, params=params, stream=True)
    request.raise_for_status()

    all_dfs = []
    total_results = 0

    while True:
        data_io = StringIO(request.text)
        try:
            df_page = pd.read_csv(data_io, sep="\t")
            all_dfs.append(df_page)
            total_results += len(df_page)
            print(f"Descargados {len(df_page)}... Total: {total_results}")

            if max_results and total_results >= max_results:
                break
        except pd.errors.EmptyDataError:
            print("Página vacía recibida.")
            break

        next_link = get_next_link(request.headers)
        if next_link:
            request = session.get(next_link, stream=True)
            request.raise_for_status()
        else:
            break

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Limpieza
        final_df.drop_duplicates(subset=["Entry"], keep="first", inplace=True)

        if max_results:
            final_df = final_df.head(max_results)

        final_df.rename(
            columns={
                "Entry": "accession",
                "Sequence": "sequence",
                "Subcellular location [CC]": "subcellular_location",
                "Organism": "organism",
            },
            inplace=True,
        )

        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        final_df.to_csv(RAW_DATA_PATH, index=False)
        print(
            f"✅ Guardados {len(final_df)} registros de múltiples especies en {RAW_DATA_PATH}"
        )
        print("Distribución por organismo:")
        print(final_df["organism"].value_counts().head())
    else:
        print("❌ No se descargaron datos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_results", type=int, default=2000, help="-1 para todo")
    args = parser.parse_args()

    limit = None if args.max_results == -1 else args.max_results
    fetch_data_with_cursor(max_results=limit)
