# example_usage.py

"""
Example usage of MSA Preprocessor
"""

from msa_preprocessor import MSAPreprocessor

def example_processing():
    """Example of running the full MSA preprocessing pipeline"""
    
    # Initialize preprocessor with custom parameters
    preprocessor = MSAPreprocessor(
        min_variant_freq=0.05,      # Merge variants with frequency < 5%
        conserved_threshold=0.95,   # Remove positions with >95% conservation
        correlation_threshold=0.95  # Merge variants with correlation > 0.95
    )
    
    # Run full preprocessing pipeline
    processed_data = preprocessor.full_preprocessing_pipeline(
        msa_file_path='path/to/your/msa_data.csv',
        output_prefix='my_msa_processed',
        save_results=True
    )
    
    print("Processing completed!")
    print(f"Final data shape: {processed_data.shape}")
    
    # Display merge summary
    merge_summary = preprocessor.get_merge_summary()
    if not merge_summary.empty:
        print("\nMerge Summary:")
        print(merge_summary.head(10))
    
    return processed_data

def process_with_defaults(msa_file_path: str):
    """Process MSA data with default parameters"""
    preprocessor = MSAPreprocessor()
    return preprocessor.full_preprocessing_pipeline(msa_file_path)

if __name__ == '__main__':
    # Run example
    example_processing()




"""
Example usage of Refined Gradient Boosting with external config
"""

from refined_gradient_boosting import RefinedGradientBoosting, Config

def example_with_config():
    """Example using external configuration file"""
    
    # Load configuration from file
    config = Config('config.json')
    
    # Initialize selector
    selector = RefinedGradientBoosting(config)
    
    # Load data
    selector.load_data(
        X_path='path/to/your/features.parquet',
        y_path='path/to/your/target.csv',
        training_samples_path='path/to/training/samples.csv'
    )
    
    # Run all stages
    features_pool = selector.feature_pooling('./results')
    print(f"Pooling completed: {len(features_pool)} features")
    
    features_distilled = selector.feature_distillation(features_pool, './results')
    print(f"Distillation completed: {len(features_distilled)} features")
    
    final_features = selector.feature_ranking(features_distilled, 50, './results')
    print(f"Final features: {len(final_features)}")
    
    return final_features

def create_custom_config():
    """Example of creating and saving a custom configuration"""
    config = Config()  # Use defaults
    config.save('my_custom_config.json')
    print("Custom configuration saved to my_custom_config.json")

if __name__ == '__main__':
    # Create a custom config file
    create_custom_config()
    
    # Run with custom config
    features = example_with_config()
    print("Selected features:")
    print(features.head(10))