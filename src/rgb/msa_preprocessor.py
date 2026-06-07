"""
MSA (Multiple Sequence Alignment) Preprocessor

This module provides functionality to preprocess MSA data for machine learning,
including one-hot encoding and filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from rgb.utils import (
    load_msa_data,
    apply_taxonomy_filter,
    filter_by_frequency,
    filter_by_ks_test,
    gene_wise_merge_variants
)

logger = logging.getLogger(__name__)


class MSAPreprocessor:
    """
    MSA Preprocessor for converting sequence alignment data to ML-ready features

    This class handles:
    - One-hot encoding of sequence variants (6 variants per position: A, T, C, G, GAP, NA)
    - Frequency-based filtering
    - Conservation filtering
    - Gene-wise correlation filtering
    """

    def __init__(
        self,
        min_variant_freq: float = 0.05,
        gene_correlation_threshold: float = 0.8,
        ks_test_p_threshold: float = 0.05,
        ks_test_min_freq: float = 0.2,
        ks_test_max_freq: float = 0.8,
        n_jobs: int = -1
    ):
        """
        Initialize MSA Preprocessor

        Args:
            min_variant_freq: Minimum frequency for variants (default: 0.05)
            gene_correlation_threshold: Gene-wise correlation threshold for merging (default: 0.8)
            ks_test_p_threshold: P-value threshold for KS test (default: 0.05, keeps features with p > 0.05)
            ks_test_min_freq: Minimum frequency of 1s for KS test (default: 0.2)
            ks_test_max_freq: Maximum frequency of 1s for KS test (default: 0.8)
            n_jobs: Number of parallel jobs (default: -1, uses 80% of CPUs; set to 1 for sequential)
        """
        self.min_variant_freq = min_variant_freq
        self.gene_correlation_threshold = gene_correlation_threshold
        self.ks_test_p_threshold = ks_test_p_threshold
        self.ks_test_min_freq = ks_test_min_freq
        self.ks_test_max_freq = ks_test_max_freq

        # Configure parallel jobs
        if n_jobs == -1:
            self.n_jobs = max(1, int(cpu_count() * 0.8))
        else:
            self.n_jobs = max(1, n_jobs)

        self.msa_data = None
        self.one_hot_data = None
        self.processed_data = None
        self.merge_dict = {}
        self.merge_summary = pd.DataFrame()

        logger.info("MSAPreprocessor initialized with parameters:")
        logger.info(f"  min_variant_freq: {min_variant_freq}")
        logger.info(f"  gene_correlation_threshold: {gene_correlation_threshold}")
        logger.info(f"  ks_test_p_threshold: {ks_test_p_threshold}")
        logger.info(f"  ks_test_min_freq: {ks_test_min_freq}")
        logger.info(f"  ks_test_max_freq: {ks_test_max_freq}")
        logger.info(f"  n_jobs: {self.n_jobs}")

    def load_data(self, msa_file_path: Union[str, Path], index_col: int = 0) -> pd.DataFrame:
        """
        Load MSA data from file

        Args:
            msa_file_path: Path to MSA CSV file
            index_col: Column to use as index (default: 0)

        Returns:
            DataFrame with MSA data
        """
        self.msa_data = load_msa_data(msa_file_path, index_col=index_col)
        logger.info(f"Loaded MSA data: {self.msa_data.shape}")
        return self.msa_data

    @staticmethod
    def _encode_single_gene(gene_data: Tuple[str, pd.Series]) -> pd.DataFrame:
        """
        Encode a single gene (for parallel processing)

        Args:
            gene_data: Tuple of (gene_name, gene_sequences)

        Returns:
            DataFrame with one-hot encoded features for this gene
        """
        gene_name, sequences = gene_data

        # Find maximum sequence length (excluding NaN)
        valid_sequences = sequences.dropna()
        if len(valid_sequences) == 0:
            return pd.DataFrame()

        seq_length = max(len(seq) for seq in valid_sequences)

        # Replace NaN with '0' * seq_length (NA placeholder)
        sequences = sequences.fillna('0' * seq_length)

        all_features = []

        # For each position in the sequence
        for pos in range(seq_length):
            # Extract characters at this position
            position_chars = sequences.apply(
                lambda seq: seq[pos].upper() if pos < len(seq) else '-'
            )

            # Map special characters: '-' -> GAP, '0' -> NA
            position_chars = position_chars.replace('-', 'GAP').replace('0', 'NA')

            # One-hot encode this position using get_dummies
            dummies = pd.get_dummies(position_chars, prefix=f"{gene_name}_{pos}")

            all_features.append(dummies)

        # Concatenate all features for this gene
        if all_features:
            return pd.concat(all_features, axis=1)
        else:
            return pd.DataFrame(index=sequences.index)

    def one_hot_encode(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Convert MSA sequences to one-hot encoded features

        Automatically detects all variants present in the sequences.
        Feature naming: GENENAME_POSITION_VARIANT

        Special handling:
        - NaN (missing sequence) → replaced with '0' (length = sequence_length)
        - '-' in sequence → GAP variant
        - '0' in sequence → NA variant
        - All other characters → kept as-is (supports both nucleotides and amino acids)

        Args:
            data: MSA DataFrame (uses self.msa_data if None)

        Returns:
            One-hot encoded DataFrame

        Example:
            For nucleotides at position 0: GENE_0_A, GENE_0_T, GENE_0_C, GENE_0_G, GENE_0_GAP, GENE_0_NA
            For amino acids at position 0: GENE_0_M, GENE_0_L, GENE_0_A, GENE_0_GAP, GENE_0_NA, ...
        """
        if data is None:
            if self.msa_data is None:
                raise ValueError("No MSA data loaded. Call load_data() first.")
            data = self.msa_data

        logger.info(f"Starting one-hot encoding with {self.n_jobs} processes...")

        # Prepare gene data for parallel processing
        gene_data_list = [(gene, data[gene].copy()) for gene in data.columns]

        if self.n_jobs != 1:
            # Parallel processing
            with Pool(processes=self.n_jobs) as pool:
                gene_features = list(tqdm(
                    pool.imap(self._encode_single_gene, gene_data_list),
                    total=len(gene_data_list),
                    desc="Encoding genes (parallel)"
                ))
        else:
            # Sequential processing (for debugging or single-core)
            gene_features = [
                self._encode_single_gene(gene_data)
                for gene_data in tqdm(gene_data_list, desc="Encoding genes")
            ]

        # Filter out empty DataFrames and concatenate
        gene_features = [gf for gf in gene_features if not gf.empty]

        if gene_features:
            self.one_hot_data = pd.concat(gene_features, axis=1)
        else:
            self.one_hot_data = pd.DataFrame(index=data.index)

        logger.info(f"One-hot encoding completed: {self.one_hot_data.shape[1]} features created")
        logger.info(f"  Total genes: {len(data.columns)}")

        return self.one_hot_data

    def filter_by_variant_frequency(
        self,
        data: Optional[pd.DataFrame] = None,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None,
        remove_gap_na: bool = True
    ) -> pd.DataFrame:
        """
        Filter features by variant frequency

        Args:
            data: Feature DataFrame (uses self.one_hot_data if None)
            min_freq: Minimum frequency (uses self.min_variant_freq if None)
            max_freq: Maximum frequency (calculated as 1 - min_freq if None)
            remove_gap_na: Whether to remove GAP and NA columns (default: True)

        Returns:
            Filtered DataFrame
        """
        if data is None:
            if self.one_hot_data is None:
                raise ValueError("No one-hot data available")
            data = self.one_hot_data

        if min_freq is None:
            min_freq = self.min_variant_freq

        if max_freq is None:
            max_freq = 1.0 - min_freq

        logger.info(f"Filtering by frequency: [{min_freq}, {max_freq}]")

        filtered_data = filter_by_frequency(
            data, min_freq=min_freq, max_freq=max_freq, n_jobs=self.n_jobs
        )

        # Remove GAP and NA columns if requested
        if remove_gap_na:
            original_cols = len(filtered_data.columns)
            gap_na_cols = [col for col in filtered_data.columns
                          if col.endswith('_GAP') or col.endswith('_NA')]
            if gap_na_cols:
                filtered_data = filtered_data.drop(columns=gap_na_cols)
                logger.info(f"Removed {len(gap_na_cols)} GAP/NA columns "
                          f"({original_cols} -> {len(filtered_data.columns)} features)")

        return filtered_data

    def filter_by_ks_test(
        self,
        data: Optional[pd.DataFrame] = None,
        p_threshold: Optional[float] = None,
        min_freq: Optional[float] = None,
        max_freq: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter features based on Kolmogorov-Smirnov test for uniform distribution

        For each binary (0/1) feature:
        1. Check if frequency of 1s is between min_freq and max_freq
        2. For positions where value == 1, test if their indices are uniformly distributed
        3. Keep features where KS test p-value > p_threshold

        Args:
            data: Feature DataFrame (uses self.one_hot_data if None)
            p_threshold: P-value threshold (uses self.ks_test_p_threshold if None)
            min_freq: Minimum frequency of 1s (uses self.ks_test_min_freq if None)
            max_freq: Maximum frequency of 1s (uses self.ks_test_max_freq if None)

        Returns:
            Filtered DataFrame
        """
        if data is None:
            if self.one_hot_data is None:
                raise ValueError("No one-hot data available")
            data = self.one_hot_data

        if p_threshold is None:
            p_threshold = self.ks_test_p_threshold

        if min_freq is None:
            min_freq = self.ks_test_min_freq

        if max_freq is None:
            max_freq = self.ks_test_max_freq

        logger.info(f"Filtering by KS test (freq: [{min_freq}, {max_freq}], p > {p_threshold})")

        filtered_data = filter_by_ks_test(
            data, p_threshold=p_threshold, min_freq=min_freq, max_freq=max_freq, n_jobs=self.n_jobs
        )

        return filtered_data

    def gene_wise_merge(
        self,
        data: Optional[pd.DataFrame] = None,
        method: str = 'max'
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Remove correlated variants within each gene (keep only the first)

        For each gene, finds groups of highly correlated features and keeps only
        the first feature from each group, removing the rest to reduce redundancy.

        Args:
            data: Feature DataFrame
            method: Deprecated - kept for backward compatibility, not used

        Returns:
            Tuple of (filtered DataFrame, merge dictionary)

            The merge dictionary maps kept features to the list of features they represent.
        """
        if data is None:
            if self.one_hot_data is None:
                raise ValueError("No one-hot data available")
            data = self.one_hot_data

        logger.info(f"Filtering correlated variants within each gene (threshold: {self.gene_correlation_threshold})")

        merged_data, merge_dict = gene_wise_merge_variants(
            data,
            correlation_threshold=self.gene_correlation_threshold,
            method=method,
            n_jobs=self.n_jobs
        )

        # Store merge information
        self.merge_dict = merge_dict

        # Create merge summary
        self._create_merge_summary(merge_dict)

        return merged_data, merge_dict

    def _create_merge_summary(self, merge_dict: dict) -> None:
        """Create summary of merge operations"""
        summary_data = []

        for merged_name, original_features in merge_dict.items():
            if len(original_features) > 1:
                summary_data.append({
                    'merged_feature': merged_name,
                    'n_merged': len(original_features),
                    'original_features': '|'.join(original_features)
                })

        self.merge_summary = pd.DataFrame(summary_data)

        if len(self.merge_summary) > 0:
            logger.info(f"Created merge summary with {len(self.merge_summary)} merge operations")

    def get_merge_summary(self) -> pd.DataFrame:
        """
        Get summary of merge operations

        Returns:
            DataFrame with merge summary
        """
        return self.merge_summary

    def full_preprocessing_pipeline(
        self,
        msa_file_path: Optional[Union[str, Path]] = None,
        taxonomy_path: Optional[Union[str, Path]] = None,
        taxonomy_filter: Optional[Dict] = None,
        output_prefix: str = 'msa_processed',
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline

        Args:
            msa_file_path: Path to MSA file (uses loaded data if None)
            taxonomy_path: Optional path to taxonomy data
            taxonomy_filter: Optional taxonomy filter dict (e.g., {'order': 'Passeriformes'})
            output_prefix: Prefix for output files (default: 'msa_processed')
            save_results: Whether to save intermediate results (default: True)

        Returns:
            Processed feature DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting MSA preprocessing pipeline")
        logger.info("=" * 60)

        # Load data if path provided
        if msa_file_path is not None:
            self.load_data(msa_file_path)

        if self.msa_data is None:
            raise ValueError("No MSA data available. Provide msa_file_path or call load_data() first.")

        # Apply taxonomy filter if provided
        data = self.msa_data
        if taxonomy_path and taxonomy_filter:
            logger.info("Applying taxonomy filter...")
            data = apply_taxonomy_filter(
                data, taxonomy_path, taxonomy_filter
            )

        # Step 1: One-hot encoding
        logger.info("\nStep 1: One-hot encoding")
        one_hot_data = self.one_hot_encode(data)

        # Step 2: Frequency filtering (removes both rare and overly common variants)
        logger.info("\nStep 2: Frequency filtering")
        filtered_data = self.filter_by_variant_frequency(one_hot_data)

        # Step 3: KS test filtering (uniform distribution test)
        logger.info("\nStep 3: KS test filtering")
        ks_filtered_data = self.filter_by_ks_test(filtered_data)

        # Step 4: Gene-wise merge (correlation filter)
        logger.info("\nStep 4: Gene-wise correlation filtering")
        processed_data, merge_dict = self.gene_wise_merge(ks_filtered_data)

        self.processed_data = processed_data

        # Save results
        if save_results:
            self._save_pipeline_results(output_prefix)

        logger.info("=" * 60)
        logger.info(f"Pipeline completed: {processed_data.shape[1]} features")
        logger.info("=" * 60)

        return processed_data

    def _save_pipeline_results(self, output_prefix: str) -> None:
        """Save pipeline results to files"""
        logger.info(f"\nSaving results with prefix: {output_prefix}")

        # Save processed features
        feature_file = f"{output_prefix}_features.csv"
        self.processed_data.to_csv(feature_file)
        logger.info(f"  Saved features: {feature_file}")

        # Save merge summary
        if len(self.merge_summary) > 0:
            summary_file = f"{output_prefix}_merge_summary.csv"
            self.merge_summary.to_csv(summary_file, index=False)
            logger.info(f"  Saved merge summary: {summary_file}")

        # Save merge dictionary
        if self.merge_dict:
            import json
            dict_file = f"{output_prefix}_merge_dict.json"
            with open(dict_file, 'w') as f:
                json.dump(self.merge_dict, f, indent=2)
            logger.info(f"  Saved merge dict: {dict_file}")


def main():
    """Command-line interface for MSA preprocessing"""
    import argparse

    parser = argparse.ArgumentParser(
        description='MSA Preprocessor - Convert MSA data to ML-ready features'
    )

    # Input/Output
    parser.add_argument('--msa_file', type=str, required=True,
                        help='Path to MSA CSV file')
    parser.add_argument('--output_prefix', type=str, default='msa_processed',
                        help='Prefix for output files')

    # Preprocessing parameters
    parser.add_argument('--min_variant_freq', type=float, default=0.05,
                        help='Minimum variant frequency (default: 0.05)')
    parser.add_argument('--gene_correlation_threshold', type=float, default=0.8,
                        help='Gene-wise correlation threshold for merging (default: 0.8)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for encoding (default: 1, use -1 for all CPUs)')

    # Taxonomy filtering
    parser.add_argument('--taxonomy_path', type=str,
                        help='Path to taxonomy CSV file')
    parser.add_argument('--taxonomy_filter', type=str,
                        help='Taxonomy filter in format "column:value"')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse taxonomy filter
    taxonomy_filter_dict = None
    if args.taxonomy_filter:
        col, val = args.taxonomy_filter.split(':')
        taxonomy_filter_dict = {col: val}

    # Initialize preprocessor
    preprocessor = MSAPreprocessor(
        min_variant_freq=args.min_variant_freq,
        gene_correlation_threshold=args.gene_correlation_threshold,
        n_jobs=args.n_jobs
    )

    # Run pipeline
    preprocessor.full_preprocessing_pipeline(
        msa_file_path=args.msa_file,
        taxonomy_path=args.taxonomy_path,
        taxonomy_filter=taxonomy_filter_dict,
        output_prefix=args.output_prefix,
        save_results=True
    )


if __name__ == '__main__':
    main()
