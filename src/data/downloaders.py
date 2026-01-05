# src/data/downloaders.py
"""
Multi-source data downloaders with robust error handling and caching.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """Abstract base class for data downloaders."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    @abstractmethod
    def download(self, **kwargs) -> pd.DataFrame:
        """Download data and return as DataFrame."""
        pass

    def _get_cache_path(self, identifier: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"{identifier}.csv"

    def _save_cache(self, df: pd.DataFrame, identifier: str):
        """Save DataFrame to cache."""
        cache_path = self._get_cache_path(identifier)
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached data to {cache_path}")

    def _load_cache(self, identifier: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if exists"""
        cache_path = self._get_cache_path(identifier)
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_path}")
            return pd.read_csv(cache_path)
        return None


class UniProtDownloader(BaseDownloader):
    """
    Download protein data from UniProt with cursor-based pagination.
    Supports multi-organism queries.
    """

    def __init__(
        self,
        cache_dir: Path,
        api_url: str = "https://rest.uniprot.org/uniprotkb/stream",
    ):
        super().__init__(cache_dir)
        self.api_url = api_url

    def download(
        self,
        organisms: List[Dict],
        fields: List[str],
        filters: List[str],
        batch_size: int = 500,
        max_results: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Download UniProt data for multiple organisms

        Args:
            organisms: List of dicts with 'taxonomy_id', 'name', 'weight'
            fields: UniProt fields to retrieve
            filters: Query filters (e.g., "reviewed:true")
            batch_size: Results per request
            max_results: Maximum total results (None for unlimited)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with downloaded protein data
        """
        cache_id = f"uniprot_multi_organism_{len(organisms)}"

        if use_cache:
            cached = self._load_cache(cache_id)
            if cached is not None:
                return cached

        all_dataframes = []

        for org in organisms:
            logger.info(f"Downloading {org['name']} (taxonomy: {org['taxonomy_id']})")

            query = self._build_query(taxonomy_id=org["taxonomy_id"], filters=filters)

            params = {
                "query": query,
                "format": "tsv",
                "fields": ",".join(fields),
                "size": batch_size,
            }

            org_df = self._fetch_with_pagination(
                params=params,
                max_results=int(max_results * org["weight"]) if max_results else None,
            )

            org_df["organism"] = org["name"]
            org_df["taxonomy_id"] = org["taxonomy_id"]
            all_dataframes.append(org_df)

            # Respect delay between organisms
            time.sleep(1)

        # Combine all organisms
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Standarize column names
        combined_df = self._standarize_columns(combined_df)

        # Remove duplicates (proteins in multiple organisms)
        combined_df.drop_duplicates(subset=["accession"], keep="first", inplace=True)

        logger.info(
            f"Downloaded {len(combined_df)} proteins from {len(organisms)} organisms"
        )

        if use_cache:
            self._save_cache(combined_df, cache_id)

        return combined_df

    def _build_query(self, taxonomy_id: int, filters: List[str]) -> str:
        """Build UniProt query string"""
        query_parts = [f"organism_id:{taxonomy_id}"]
        query_parts.extend(filters)
        return " AND ".join([f"{part}" for part in query_parts])

    def _fetch_with_pagination(
        self, params: Dict, max_results: Optional[int]
    ) -> pd.DataFrame:
        """Fetch data using cursor-based pagination"""
        all_data = []
        total_fetched = 0

        response = self.session.get(self.api_url, params=params, stream=True)
        response.raise_for_status()

        with tqdm(desc="Downloading", unit=" proteins") as pbar:
            while True:
                try:
                    data_io = StringIO(response.text)
                    df_page = pd.read_csv(data_io, sep="\t")

                    all_data.append(df_page)
                    total_fetched += len(df_page)
                    pbar.update(len(df_page))

                    if max_results and total_fetched >= max_results:
                        break

                    # Check for next page
                    next_link = self._get_next_link(response.headers)
                    if not next_link:
                        break

                    response = self.session.get(next_link, stream=True)
                    response.raise_for_status()

                except pd.errors.EmptyDataError:
                    logger.warning("Empty page received, stopping pagination")
                    break
                except Exception as e:
                    logger.error(f"Error during pagination: {e}")
                    break

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _get_next_link(self, headers: Dict) -> Optional[str]:
        """Extract next page URL from Link header"""
        if "Link" in headers:
            match = re.search(r'<(.+?); rel="next"', headers["Link"])
            if match:
                return match.group(1)
        return None

    def _standarize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standarize UniProt column names"""
        rename_map = {
            "Entry": "accession",
            "Sequence": "sequence",
            "Subcellular location [CC]": "subcellular_location",
            "Organism": "organism_name",
            "Gene Names": "gene_names",
            "Length": "sequence_length",
        }

        df.rename(columns=rename_map, inplace=True)
        return df


class YeastGenomeDownloader(BaseDownloader):
    """Download additional yeast data from SGD"""

    def __init__(
        self, cache_dir: Path, api_url: "https://www.yeastgenome.org/webservice"
    ):
        super().__init__(cache_dir)
        self.api_url = api_url

    def download(self, gene_names: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Download detailed yeast protein information from SGD

        Args:
            gene_names: List of yeast gene names
            use_cache: Whether to use cached data

        Returns:
            DataFrame with SGD annotations
        """
        cache_id = "sgd_annotations"

        if use_cache:
            cached = self._load_cache(cache_id)
            if cached is not None:
                return cached

        annotations = []

        for gene in tqdm(gene_names, desc="Querying SGD"):
            try:
                response = self.session.get(
                    f"{self.api_url}/locus/{gene}/protein_experiment_details",
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    annotations.append(
                        {
                            "gene_name": gene,
                            "localization_experiments": len(
                                data.get("localization", [])
                            ),
                            "go_terms": [
                                go["go"]["display_name"]
                                for go in data.get("go_annotations", [])
                            ],
                            "phenotypes": [
                                p["phenotype"]["display_name"]
                                for p in data.get("phenotypes", [])
                            ],
                        }
                    )

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Failed to fetch {gene}: {e}")
                continue

        df = pd.DataFrame(annotations)

        if use_cache:
            self._save_cache(df, cache_id)

        return df


class YPLDownloader(BaseDownloader):
    """Download Yeast Protein Localization database"""

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir)

    def download(
        self, url: str = "https://ypl.epfl.ch/data/YPL.db2", use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download YPL database

        Args:
            url: YPL download URL
            use_cache: Whether to use cached data

        Returns:
            DataFrame with YPL localization data
        """
        cache_id = "ypl_database"

        if use_cache:
            cached = self._load_cache(cache_id)
            if cached is not None:
                return cached

        logger.info("Downloading YPL database")

        # YPL is typically distributed as a text file
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        # Parse YPL format (custom format, adjust as needed)
        data = self._parse_ypl_format(response.text)
        df = pd.DataFrame(data)

        if use_cache:
            self._save_cache(df, cache_id)

        return df

    def _parse_ypl_format(self, text: str) -> List[Dict]:
        """Parse YPL custom format"""
        # Implement parsing logic based on actual YPL format
        # This is a placeholder
        data = []

        for line in text.split("\n"):
            if line.strip() and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    data.append(
                        {
                            "gene_name": parts[0],
                            "localization": parts[1],
                            "confidence": parts[2],
                        }
                    )

        return data


class DataDownloaderManager:
    """Orchestrates downloads from multiple sources"""

    def __init__(self, config: Dict, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir

        self.uniprot = UniProtDownloader(
            cache_dir=self.cache_dir / "uniprot", api_url=config["uniprot"]["api_url"]
        )

        self.sgd = YeastGenomeDownloader(
            cache_dir=self.cache_dir / "sgd",
            api_url=config.get("yeastgenome", {}).get("api_url"),
        )

        self.ypl = YPLDownloader(cache_dir=self.cache_dir / "ypl")

    def download_all(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Download and integrate data from all sources

        Returns:
            Unified DataFrame with data from all sources
        """
        logger.info("Starting multi-source data download...")

        # 1. Download UniProt (primary source)
        uniprot_data = self.uniprot.download(
            organisms=self.config["uniprot"]["organisms"],
            fields=self.config["uniprot"]["fields"],
            filters=self.config["uniprot"]["query_filters"],
            batch_size=self.config["uniprot"]["batch_size"],
            max_results=self.config["uniprot"].get("max_results"),
            use_cache=use_cache,
        )

        # 2. Enrich yeast data with SGD
        yeast_data = uniprot_data[uniprot_data["taxonomy_id"] == 559292]
        if not yeast_data.empty:
            gene_names = yeast_data["gene_names"].dropna().str.split().str[0].tolist()
            sgd_data = self.sgd.download(
                gene_names=gene_names[:1000], use_cache=use_cache
            )

            # Merge SGD annotations
            uniprot_data = uniprot_data.merge(
                sgd_data, left_on="gene_names", right_on="gene_name", how="left"
            )

        # 3. Integrate YPL data
        try:
            ypl_data = self.ypl.download(use_cache=use_cache)
            # Merge YPL localization confidence scores
            uniprot_data = uniprot_data.merge(
                ypl_data,
                left_on="gene_names",
                right_on="gene_name",
                how="left",
                suffixes=("", "_ypl"),
            )
        except Exception as e:
            logger.warning(f"Could not download YPL data: {e}")

        logger.info(f"Total proteins after integration: {len(uniprot_data)}")

        return uniprot_data
