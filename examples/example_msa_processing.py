"""
Example: MSA Preprocessing Pipeline

This example demonstrates how to use the MSAPreprocessor to convert
Multiple Sequence Alignment data into ML-ready features.
"""

from rgb.msa_preprocessor import MSAPreprocessor
import logging
import json
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def example_basic_processing():
    """Basic MSA preprocessing with default parameters"""

    # Initialize preprocessor (default uses 80% of CPUs)
    preprocessor = MSAPreprocessor()

    # Run full pipeline
    processed_data = preprocessor.full_preprocessing_pipeline(
        msa_file_path='data/your_msa_data.csv',
        output_prefix='output/msa_processed',
        save_results=True
    )

    print(f"\nProcessing completed!")
    print(f"Output shape: {processed_data.shape}")

    return processed_data


def example_custom_parameters():
    """MSA preprocessing with custom parameters"""

    # Initialize with custom parameters
    preprocessor = MSAPreprocessor(
        min_variant_freq=0.10,           # Keep variants with frequency > 10%
        gene_correlation_threshold=0.85, # Gene-wise merge threshold
        n_jobs=8                         # Use 8 parallel processes
    )

    # Run pipeline
    processed_data = preprocessor.full_preprocessing_pipeline(
        msa_file_path='data/your_msa_data.csv',
        output_prefix='output/msa_custom',
        save_results=True
    )

    # Check merge summary
    merge_summary = preprocessor.get_merge_summary()
    if not merge_summary.empty:
        print("\nMerge Summary:")
        print(merge_summary.head())

    return processed_data


def example_with_taxonomy_filter():
    """MSA preprocessing with taxonomy filtering"""

    preprocessor = MSAPreprocessor(
        min_variant_freq=0.05,
        gene_correlation_threshold=0.8,
        n_jobs=-1  # Use 80% of all available CPUs
    )

    # Run pipeline with taxonomy filter
    processed_data = preprocessor.full_preprocessing_pipeline(
        msa_file_path='data/your_msa_data.csv',
        taxonomy_path='data/taxonomy.csv',
        taxonomy_filter={'order': 'Passeriformes'},  # Filter by taxonomic order
        output_prefix='output/msa_passeriformes',
        save_results=True
    )

    return processed_data


def example_step_by_step():
    """Step-by-step MSA preprocessing for more control"""

    preprocessor = MSAPreprocessor(
        min_variant_freq=0.10,
        gene_correlation_threshold=0.8,
        ks_test_p_threshold=0.05,
        ks_test_min_freq=0.2,
        ks_test_max_freq=0.8,
        n_jobs=-1  # Use 80% of all available CPUs
    )

    # Step 1: Load data
    # msa_data = preprocessor.load_data('/vepfs-mlp2/mlp-public/liqiuyi/gener_v1/bird_beak/MSA_test.csv')
    msa_data = preprocessor.load_data('/vepfs-mlp2/mlp-public/liqiuyi/gener_v1/bird_beak/Beak_Seq_aln_1034.csv')
    # msa_data.index = msa_data.index.str.replace('Anser_caerulscens', 'Anser_caerulescens', regex=False)
    print(f"Loaded MSA data: {msa_data.shape}")

    # species_info = pd.read_csv('/vepfs-mlp2/mlp-public/liqiuyi/gener_v1/bird_beak/morph_data_1034.csv', index_col=0)
    # msa_data = msa_data.loc[species_info.index]
    # msa_data = msa_data[~msa_data.index.duplicated(keep='first')]
    # msa_data.to_csv('/vepfs-mlp2/mlp-public/liqiuyi/gener_v1/bird_beak/Beak_Seq_aln_1034.csv')

    # Step 2: One-hot encode
    one_hot_data = preprocessor.one_hot_encode()
    print(f"One-hot encoded: {one_hot_data.shape}")

    # Step 3: Filter by frequency (removes both rare and common variants)
    filtered_data = preprocessor.filter_by_variant_frequency(one_hot_data)
    print(f"After frequency filter: {filtered_data.shape}")

    # Step 4: KS test filtering
    ks_filtered_data = preprocessor.filter_by_ks_test(filtered_data)
    print(f"After KS test filter: {ks_filtered_data.shape}")

    # Step 5: Gene-wise correlation filtering (keep first, remove correlated)
    final_data, merge_dict = preprocessor.gene_wise_merge(ks_filtered_data)
    print(f"After gene-wise correlation filter: {final_data.shape}")
    print("Note: Highly correlated features within each gene were removed (kept first only)")

    # Save results
    final_data.to_parquet('./X_genome_filtered.parquet', index=False)
    print("Saved features to X_genome_filtered.parquet")

    # # Save merge dictionary
    # with open('./merge_dict.json', 'w') as f:
    #     json.dump(merge_dict, f, indent=2)
    # print(f"Saved merge dictionary with {len(merge_dict)} entries")

    # # Save merge summary
    # merge_summary = preprocessor.get_merge_summary()
    # if not merge_summary.empty:
    #     merge_summary.to_csv('./merge_summary.csv', index=False)
    #     print(f"Saved merge summary: {len(merge_summary)} merge operations")

    return final_data


if __name__ == '__main__':
    print("=" * 70)
    print("MSA Preprocessing Examples")
    print("=" * 70)

    # Choose which example to run
    # Uncomment the one you want to use

    # Example 1: Basic processing with defaults
    # processed_data = example_basic_processing()

    # Example 2: Custom parameters
    # processed_data = example_custom_parameters()

    # Example 3: With taxonomy filter
    # processed_data = example_with_taxonomy_filter()

    # Example 4: Step-by-step processing
    processed_data = example_step_by_step()

    print("\nNote: Uncomment one of the examples above to run it.")
    print("Make sure to update file paths to point to your actual data files.")
