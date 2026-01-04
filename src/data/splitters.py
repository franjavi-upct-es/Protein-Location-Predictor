# src/data/splitters.py
"""
Homology-aware data splitting using GraphPart to prevent data leakage.
"""

import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
import tempfile
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


class GraphPartSplitter:
    """
    Split protein sequences using GraphPart algorithm.
    Ensures no sequence in test set has >threshold similarity to training set
    """

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        coverage_threshold: float = 0.80,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ):
        """
        Args:
            similarity_threshold: Maximum sequence identity between partitions (0-1)
            coverage_threshold: Minimum alignment coverage for MMseqs2 (0-1)
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
        """
        self.similarity_threshold = similarity_threshold
        self.coverage_threshold = coverage_threshold
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1.0 - test_size - val_size

        # Verify tools are installed
        self._check_dependencies()

    def _check_dependencies(self):
        """Verify required tools are installed"""
        try:
            subprocess.run(["mmseqs", "version"], capture_output=True, check=True)
            logger.info("✓ MMseqs2 found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise EnvironmentError(
                "MMseqs2 not found. Install with: conda install -c bioconda mmseqs2"
            )

        try:
            subprocess.run(
                ["graphpart", "--help"],
                capture_output=True,
                check=True,
            )
            logger.info("✓ GraphPart found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise EnvironmentError(
                "GraphPart not found. Install with: pip install graph-part"
            )

    def split(
        self,
        df: pd.DataFrame,
        sequence_column: str = "sequence",
        id_column: str = "accession",
        stratify_column: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame into train/val/test sets using GraphPart

        Args:
            df: DataFrame with protein sequences
            sequence_column: Name of column containing sequences
            id_column: Name of column containing unique IDs
            stratify_column: Column to maintain class distribution (optional)
            output_dir: Directory to save intermediate files (use temp if None)

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing DataFrames
        """
        logger.info(f"Starting GraphPart split with {len(df)} sequences")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(
            f"Target splits - Train: {self.train_size:.1%}, Val: {self.val_size:.1%}, Test: {self.test_size:.1%}"
        )

        # Use temporary directory if not specified
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="graphpart_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Working directory: {output_dir}")

        # Step 1: Write sequences to FASTA
        fasta_path = output_dir / "sequences.fasta"
        self._write_fasta(df, fasta_path, sequence_column, id_column)

        # Step 2: Run MMseqs2 for all-vs-all similarity
        similarity_tsv = self._run_mmseqs2(fasta_path, output_dir)

        # Step 3: Run GraphPart for optimal partitioning
        partition_file = self._run_graphpart(similarity_tsv, output_dir)

        # Step 4: Parse partitions and create split DataFrames
        splits = self._create_splits(df, partition_file, id_column, stratify_column)

        # Step 5: Validate splits
        self._validate_splits(splits, df)

        logger.info("✓ GraphPart splitting complete")

        return splits

    def _write_fasta(
        self, df: pd.DataFrame, fasta_path: Path, sequence_column: str, id_column: str
    ):
        """Write DataFrame sequences to FASTA file"""
        logger.info(f"Writting sequences to {fasta_path}")

        records = []
        for _, row in df.iterrows():
            record = SeqRecord(
                Seq(row[sequence_column]), id=str(row[id_column]), description=""
            )
            records.append(record)

        SeqIO.write(records, fasta_path, "fasta")
        logger.info(f"Wrote {len(records)} sequences")

    def _run_mmseqs2(self, fasta_path: Path, output_dir: Path) -> Path:
        """
        Run MMseqs2 for all-vs-all sequence comparison.
        Returns path to similarity TSV file.
        """
        logger.info("Running MMseqs2 for sequence similarity search...")

        db_path = output_dir / "seqdb"
        result_db = output_dir / "resultdb"
        tmp_dir = output_dir / "tmp"
        tsv_path = output_dir / "similarities.tsv"

        tmp_dir.mkdir(exist_ok=True)

        # Create database
        logger.info("  Creating MMseqs2 database...")
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_path)],
            check=True,
            capture_output=True,
        )

        # All-vs-all search
        logger.info("  Running all-vs-all search...")
        subprocess.run(
            [
                "mmseqs",
                "search",
                str(db_path),
                str(db_path),
                str(result_db),
                str(tmp_dir),
                "--min-seq-id",
                str(self.similarity_threshold),
                "-c",
                str(self.coverage_threshold),
                "--cov-mode",
                "0",  # Coverage of query
                "-s",
                "7.5",  # Sensitivity
                "--threads",
                "4",
            ],
            check=True,
            capture_output=True,
        )

        # Convert to TSV
        logger.info("  Converting results to TSV...")
        subprocess.run(
            [
                "mmseqs",
                "convertails",
                str(db_path),
                str(db_path),
                str(result_db),
                str(tsv_path),
                "--format-output",
                "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tend,evaluate,bits",
            ],
            check=True,
            capture_output=True,
        )

        # Count similarities
        with open(tsv_path) as f:
            num_similarities = sum(1 for _ in f)

        logger.info(f"  Found {num_similarities} sequence pairs above threshold")

        return tsv_path

    def _run_graphpart(self, similarity_tsv: Path, output_dir: Path) -> Path:
        """
        Run GraphPart algorithm for optimal data partitioning.
        Returns path to partition assigment file.
        """
        logger.info("Running GraphPart for optimal partitioning...")

        partition_file = output_dir / "partitions.txt"

        # Prepare partition sizes
        partition_sizes = (
            f"{self.train_size:.4f},{self.val_size:.4f},{self.test_size:.4f}"
        )

        cmd = [
            "graph-part",
            "--similarity-file",
            str(similarity_tsv),
            "--output",
            str(partition_file),
            "--num-partitions",
            "3",
            "--partition-sizes",
            partition_sizes,
            "--threshold",
            str(self.similarity_threshold),
            "--algorithm",
            "kernighan-lin",  # K-L algorithm for balanced cuts
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("  GraphPart completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"GraphPart failed: {e.stderr}")
            raise

        return partition_file

    def _create_splits(
        self,
        df: pd.DataFrame,
        partition_file: Path,
        id_column: str,
        stratify_column: Optional[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train/val/test DataFrames from GraphPart assigments.
        """
        logger.info("Creating dataset splits from GraphPart assigments...")

        # Parse partition assigments
        partitions = {}
        with open(partition_file) as f:
            for line in f:
                if line.strip():
                    seq_id, partition_idx = line.strip().split("\t")
                    partitions[seq_id] = int(partition_idx)

        # Create splits
        df = df.copy()
        df["partition"] = df[id_column].astype(str).map(partitions)

        splits = {
            "train": df[df["partition"] == 0].drop("partition", axis=1),
            "val": df[df["partition"] == 1].drop("partition", axis=1),
            "test": df[df["partition"] == 2].drop("partition", axis=1),
        }

        # Log split statistics
        for split_name, split_df in splits.items():
            logger.info(
                f"  {split_name}: {len(split_df)} samples ({len(split_df) / len(df) * 100:.1f}%)"
            )

            if stratify_column and stratify_column in split_df.columns:
                class_dist = split_df[stratify_column].value_counts()
                logger.info(f"  Class distribution: {class_dist.to_dict()}")

        return splits

    def _validate_splits(
        self, splits: Dict[str, pd.DataFrame], original_df: pd.DataFrame
    ):
        """
        Validate that splits are complete and non-overlapping.
        """
        total_samples = sum(len(df) for df in splits.values())

        if total_samples != len(original_df):
            logger.warning(
                f"Sample count mismatch: {total_samples} in splits vs {len(original_df)} in original"
            )

        # Check for overlap (shouldn't happen with GraphPart)
        train_ids = set(splits["train"]["accession"])
        val_ids = set(splits["val"]["accession"])
        test_ids = set(splits["test"]["accession"])

        assert len(train_ids & val_ids) == 0, "Train/Val overlap detected!"
        assert len(train_ids & test_ids) == 0, "Train/Test overlap detected!"
        assert len(val_ids & test_ids) == 0, "Val/Test overlap detected!"

        logger.info("✓ Splits validated: No overlap detected")


class RandomSplitter:
    """
    Baseline random splitter for comparison.
    Stratified by class labels.
    """

    def __init__(
        self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, stratify_column: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform stratified random split (baseline method).

        Args:
            df: DataFrame to split
            stratify_column: Column to maintain distribution

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        from sklearn.model_selection import train_test_split

        stratify = df[stratify_column] if stratify_column else None

        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # Second split: train vs val
        if stratify is not None:
            stratify_train_val = train_val[stratify_column]
        else:
            stratify_train_val = None

        train, val = train_test_split(
            train_val,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=stratify_train_val,
        )

        logger.info(
            f"Random split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
        )

        return {"train": train, "val": val, "test": test}


def create_splitter(method: str, config: Dict):
    """
    Factory function to create appropriate splitter.

    Args:
        method: 'graphpart' or 'random'
        config: Configuration dictionary

    Returns:
        Splitter instance
    """
    if method == "graphpart":
        return GraphPartSplitter(
            similarity_threshold=config.get("similarity_threshold", 0.30),
            test_size=config.get("test_size", 0.15),
            val_size=config.get("val_size", 0.15),
        )
    elif method == "random":
        return RandomSplitter(
            test_size=config.get("test_size", 0.15),
            val_size=config.get("val_size", 0.15),
        )
    else:
        raise ValueError(f"Unknown splitting method: {method}")
