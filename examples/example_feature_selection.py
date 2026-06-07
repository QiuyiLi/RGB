"""
Example: Refined Gradient Boosting Feature Selection

This example demonstrates how to use the RefinedGradientBoosting class
for feature selection on high-dimensional data.
"""

from rgb.feature_selection import RefinedGradientBoosting, Config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def example_basic_feature_selection():
    """Basic feature selection with default configuration"""

    # Load configuration from file
    config = Config('../configs/config.json')

    # Initialize feature selector
    selector = RefinedGradientBoosting(config)

    # Load data
    selector.load_data(
        X_path='data/features.parquet',
        y_path='data/target.csv'
    )

    # Run all three stages
    print("\n=== Stage 1: Feature Pooling ===")
    features_pool = selector.feature_pooling('./results')
    print(f"Pooling completed: {len(features_pool)} features")

    print("\n=== Stage 2: Feature Distillation ===")
    features_distilled = selector.feature_distillation(features_pool, './results')
    print(f"Distillation completed: {len(features_distilled)} features")

    print("\n=== Stage 3: Feature Ranking ===")
    final_features = selector.feature_ranking(features_distilled, 50, './results')
    print(f"Final ranking completed: {len(final_features)} features")

    return final_features


def example_with_taxonomy_filter():
    """Feature selection with taxonomy filtering"""

    config = Config('../configs/config.json')
    selector = RefinedGradientBoosting(config)

    # Load data with taxonomy filter
    taxonomy_filter = {
        'path': 'data/taxonomy.csv',
        'filters': {'order': 'Passeriformes'},
        'index_col': 'species_name'
    }

    selector.load_data(
        X_path='data/features.parquet',
        y_path='data/target.csv',
        taxonomy_filter=taxonomy_filter
    )

    # Run feature selection
    features_pool = selector.feature_pooling('./results/passeriformes')
    features_distilled = selector.feature_distillation(features_pool, './results/passeriformes')
    final_features = selector.feature_ranking(features_distilled, 50, './results/passeriformes')

    return final_features


def example_single_stage():
    """Run only specific stages of feature selection"""

    config = Config('../configs/config.json')
    selector = RefinedGradientBoosting(config)

    selector.load_data(
        X_path='data/features.parquet',
        y_path='data/target.csv'
    )

    # Run only pooling stage
    print("Running only pooling stage...")
    features_pool = selector.feature_pooling('./results')

    # Save intermediate results
    import pandas as pd
    pd.DataFrame({'feature': features_pool}).to_csv('./results/pooled_features.csv')

    return features_pool


def example_custom_config():
    """Create and use custom configuration"""

    # Create custom configuration
    config = Config()  # Start with defaults

    # Modify parameters programmatically
    config._config['model_params']['learning_rate'] = 0.01
    config._config['feature_pooling']['step'] = 100
    config._config['feature_pooling']['n_iter'] = 20

    # Save custom config
    config.save('../configs/custom_config.json')

    # Use the custom config
    selector = RefinedGradientBoosting(config)

    selector.load_data(
        X_path='data/features.parquet',
        y_path='data/target.csv'
    )

    features_pool = selector.feature_pooling('./results/custom')

    return features_pool


def example_with_training_samples():
    """Feature selection with predefined training samples"""

    config = Config('../configs/quick_test_config.json')
    selector = RefinedGradientBoosting(config)

    # Load data with specific training samples
    selector.load_data(
        X_path='/vepfs-mlp2/mlp-public/liqiuyi/bird_beak/X_genome_1034_corr0.8.parquet',
        y_path='/vepfs-mlp2/mlp-public/liqiuyi/bird_beak/res_beak_depth_1034.csv',
        training_samples_path=None
        # training_samples_path='/vepfs-mlp2/mlp-public/liqiuyi/bird_beak/training_sample_1034_1.csv'
    )

    # Run complete pipeline
    features_pool = selector.feature_pooling('./results')
    features_distilled = selector.feature_distillation(features_pool, './results')
    final_features = selector.feature_ranking(features_distilled, 30, './results')

    return final_features


if __name__ == '__main__':
    print("=" * 70)
    print("Refined Gradient Boosting Feature Selection Examples")
    print("=" * 70)

    # Choose which example to run
    # Uncomment the one you want to use

    # Example 1: Basic feature selection
    # final_features = example_basic_feature_selection()

    # Example 2: With taxonomy filter
    # final_features = example_with_taxonomy_filter()

    # Example 3: Single stage only
    # features = example_single_stage()

    # Example 4: Custom configuration
    # features = example_custom_config()

    # Example 5: With training samples
    # final_features = example_with_training_samples()

    print("\nNote: Uncomment one of the examples above to run it.")
    print("Make sure to update file paths to point to your actual data files.")
