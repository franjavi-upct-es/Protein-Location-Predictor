# src/data/download.py
"""
Multi-species protein data download from UniProt.

Downloads reviewed (Swiss-Prot) proteins with subcellular location
annotations from multiple organisms using the UniProt REST API with
cursor-based pagination.

Usage::

    from src.data.download import download_proteins

CLI:
    uv run python -m src.data.download
    uv run python -m src.data.download --max-results 1000
"""

from __future__ import annotations

import argparse
import re
from io import StringIO

import pandas as pd
import requests

from src.utils.config import DotDict, load_config, resolve_path
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# UniProt API interaction
# ---------------------------------------------------------------------------

_DEFAULT_FIELDS = "accession,sequence,cc_subcellular_location,organism_id,organism_name"


def _get_next_link(headers: dict) -> str | None:
    """Extract the next-page URL from UniProt's Link header."""
    link = headers.get("Link", "")
    match = re.search(r'<(.+?)>; rel="next"', link)
    return match.group(1) if match else None


def _download_organism(
    session: requests.Session,
    api_url: str,
    organism_id: int,
    reviewed: bool,
    fields: str,
    max_results: int | None = None,
) -> pd.DataFrame:
    """
    Download proteins for a single organism with cursor-based pagination.

    Args:
        session: Reusable requests session.
        api_url: UniProt REST API streaming endpoint.
        organism_id: NCBI taxonomy ID (e.g. 9606 for human).
        reviewed: If True, only download Swiss-Prot (reviewed) entries.
        fields: Comma-separated list of UniProt fields to retrieve.
        max_results: Optional cap on the number of proteins to download.

    Returns:
        DataFrame with columns matching the requested fields.
    """
    reviewed_str = "true" if reviewed else "false"
    params = {
        "query": (
            f"(organism_id:{organism_id}) "
            f"AND (reviewed:{reviewed_str}) "
            f"AND (cc_subcellular_location:*)"
        ),
        "format": "tsv",
        "fields": fields,
    }

    logger.info(f"Downloading organism {organism_id} (reviewed={reviewed})...")
    response = session.get(api_url, params=params, stream=True)
    response.raise_for_status()

    all_pages: list[pd.DataFrame] = []
    total = 0

    while True:
        page_df = pd.read_csv(StringIO(response.text), sep="\t")
        if page_df.empty:
            break

        all_pages.append(page_df)
        total += len(page_df)
        logger.debug(f"  Organism {organism_id}: {total} proteins downloaded so far")

        if max_results and total >= max_results:
            logger.info(f"  Reached max_results limit ({max_results}) for organism {organism_id}")
            break

        next_url = _get_next_link(dict(response.headers))
        if not next_url:
            break

        response = session.get(next_url, stream=True)
        response.raise_for_status()

    if not all_pages:
        logger.warning(f"  No data downloaded for organism {organism_id}")
        return pd.DataFrame()

    result = pd.concat(all_pages, ignore_index=True)

    if max_results and len(result) > max_results:
        result = result.head(max_results)

    logger.info(f"  Organism {organism_id}: {len(result)} proteins downloaded")
    return result


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

# UniProt column names vary by API version; normalize to our internal schema
_COLUMN_MAP = {
    "Entry": "accession",
    "From": "accession",
    "Sequence": "sequence",
    "Subcellular location [CC]": "subcellular_location",
    "Organism (ID)": "organism_id",
    "Organism": "organism_name",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UniProt TSV columns to our internal schema."""
    renamed = df.rename(columns=_COLUMN_MAP)

    # Verify required columns exists
    required = {"accession", "sequence", "subcellular_location"}
    missing = required - set(renamed.columns)
    if missing:
        available = set(renamed.columns)
        raise ValueError(
            f"Missing required columns after renaiming: {missing}. Available columns: {available}"
        )

    return renamed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_proteins(
    cfg: DotDict,
    max_results_per_organism: int | None = None,
) -> pd.DataFrame:
    """
    Download multi-species protein data from UniProt.

    Args:
        cfg: Project configuration (from load_config)
        max_results_per_organism: Optional cap per organism. Overrides config.

    Returns:
        Combined DataFrame with all downloaded proteins, deduplicated.
    """
    data_cfg = cfg.data
    api_url = data_cfg.uniprot_api_url
    fields = data_cfg.get("fields", _DEFAULT_FIELDS)
    sources = data_cfg.sources

    session = requests.Session()
    session.headers.update({"Accept": "text/plain"})

    all_dfs: list[pd.DataFrame] = []

    for source in sources:
        organism_id = source["organism_id"] if isinstance(source, dict) else source.organism_id
        reviewed = (
            source.get("reviewed", True)
            if isinstance(source, dict)
            else getattr(source, "reviewed", True)
        )

        df = _download_organism(
            session=session,
            api_url=api_url,
            organism_id=organism_id,
            reviewed=reviewed,
            fields=fields,
            max_results=max_results_per_organism,
        )
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("No data downloaded from any organism source.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = _normalize_columns(combined)

    # Deduplicated by accession
    before = len(combined)
    combined.drop_duplicates(subset=["accession"], keep="first", inplace=True)
    dupes = before - len(combined)
    if dupes > 0:
        logger.info(f"Removed {dupes} duplicate accessions")

    logger.info(
        f"Download complete: {len(combined)} unique proteins from {len(sources)} organism(s)"
    )

    # Save raw data
    raw_dir = resolve_path(cfg, "paths.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "uniprot_data.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Raw data saved to {output_path}")

    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the download CLI command."""
    parser = argparse.ArgumentParser(description="Download protein data from UniProt.")
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Maximum results per organism. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        help="Config overrides in key=value format.",
    )
    args = parser.parse_args()

    cfg = load_config(overrides=args.config_overrides)
    setup_logging(level=cfg.project.log_level)

    max_results = None if args.max_results == -1 else args.max_results
    download_proteins(cfg, max_results_per_organism=max_results)


if __name__ == "__main__":
    main()
