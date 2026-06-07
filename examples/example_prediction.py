"""
Example: Model Training and Evaluation with Selected Features

This example demonstrates how to use RGBPredictor to train and evaluate
prediction models using features selected by RGB feature selection.
"""

from rgb import RGBPredictor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def example_basic_prediction():
    """Basic prediction with cross-validation and test evaluation"""

    # Initialize predictor for regression task
    predictor = RGBPredictor(
        task='regression',
        n_folds=10,
        test_size=0.2,
        random_state=42
    )

    # Load data with selected features
    predictor.load_data(
        X='data/features.parquet',
        y='data/target.csv',
        selected_features='results/feature_ranking.csv'  # From RGB feature selection
    )

    # Run 10-fold cross-validation
    cv_results = predictor.cross_validate()
    print(f"\nCV R² (mean ± std): {cv_results['val_r2_mean']:.4f} ± {cv_results['val_r2_std']:.4f}")

    # Train on full train set and evaluate on test set
    test_results = predictor.train_test_evaluate()
    print(f"Test R²: {test_results['val_r2']:.4f}")

    # Save results
    predictor.save_results('./prediction_results')

    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    print(f"\nTop 10 important features:")
    print(feature_importance.head(10))

    return predictor


def example_classification():
    """Binary classification example"""

    # Initialize predictor for classification
    predictor = RGBPredictor(
        task='classification',
        n_folds=5,
        test_size=0.2,
        random_state=42
    )

    # Load data
    predictor.load_data(
        X='data/features.csv',
        y='data/labels.csv',
        selected_features='results/top_features.txt',
        target_column='label'
    )

    # Cross-validation
    cv_results = predictor.cross_validate()
    print(f"\nCV Accuracy: {cv_results['val_accuracy_mean']:.4f} ± {cv_results['val_accuracy_std']:.4f}")
    print(f"CV AUC: {cv_results['val_auc_mean']:.4f} ± {cv_results['val_auc_std']:.4f}")

    # Test evaluation
    test_results = predictor.train_test_evaluate()
    print(f"\nTest Accuracy: {test_results['val_accuracy']:.4f}")
    print(f"Test AUC: {test_results['val_auc']:.4f}")

    # Save results
    predictor.save_results('./classification_results', prefix='classification')

    return predictor


def example_custom_model_params():
    """Prediction with custom LightGBM parameters"""

    # Custom model parameters
    custom_params = {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'num_leaves': 50,
        'max_depth': 8,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    predictor = RGBPredictor(
        task='regression',
        n_folds=10,
        model_params=custom_params
    )

    predictor.load_data(
        X='data/features.parquet',
        y='data/target.csv',
        selected_features='results/feature_ranking.csv'
    )

    # Run evaluation
    cv_results = predictor.cross_validate(early_stopping_rounds=30)
    test_results = predictor.train_test_evaluate(early_stopping_rounds=30)

    predictor.save_results('./custom_model_results')

    return predictor


def example_without_feature_selection():
    """Use all features without prior feature selection"""

    predictor = RGBPredictor(
        task='regression',
        n_folds=10
    )

    # Load data without selected_features parameter (use all features)
    predictor.load_data(
        X='data/features.parquet',
        y='data/target.csv'
    )

    print(f"Using all {predictor.X.shape[1]} features")

    # Evaluate
    cv_results = predictor.cross_validate()
    test_results = predictor.train_test_evaluate()

    predictor.save_results('./all_features_results')

    return predictor


def example_cv_only():
    """Only run cross-validation without final test evaluation"""

    predictor = RGBPredictor(
        task='regression',
        n_folds=10
    )

    predictor.load_data(
        X='data/features.parquet',
        y='data/target.csv',
        selected_features='results/feature_ranking.csv'
    )

    # Only cross-validation
    cv_results = predictor.cross_validate()

    # Get detailed results for each fold
    cv_df = predictor.get_cv_results()
    print("\nPer-fold results:")
    print(cv_df[['fold', 'val_r2', 'val_rmse', 'val_corr']])

    predictor.save_results('./cv_only_results')

    return predictor


def example_complete_pipeline():
    """Complete pipeline: feature selection → prediction"""

    from rgb import RefinedGradientBoosting, Config

    # Step 1: Feature selection
    print("="*70)
    print("Step 1: Feature Selection")
    print("="*70)

    config = Config()
    selector = RefinedGradientBoosting(config)

    selector.load_data(
        X_path='data/features.parquet',
        y_path='data/target.csv'
    )

    # Run feature selection (abbreviated for example)
    features_pool = selector.feature_pooling('./results')
    features_distilled = selector.feature_distillation(features_pool, './results')
    final_features = selector.feature_ranking(features_distilled, 30, './results')

    selected_feature_names = final_features['feature'].tolist()
    print(f"\nSelected {len(selected_feature_names)} features")

    # Step 2: Model training and evaluation
    print("\n" + "="*70)
    print("Step 2: Model Training and Evaluation")
    print("="*70)

    predictor = RGBPredictor(
        task='regression',
        n_folds=10,
        test_size=0.2
    )

    # Use the selected features
    predictor.load_data(
        X='data/features.parquet',
        y='data/target.csv',
        selected_features=selected_feature_names  # Can pass list directly
    )

    # Evaluate
    cv_results = predictor.cross_validate()
    test_results = predictor.train_test_evaluate()

    # Save everything
    predictor.save_results('./complete_pipeline_results')

    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)

    return selector, predictor


if __name__ == '__main__':
    print("=" * 70)
    print("RGB Prediction Examples")
    print("=" * 70)

    # Choose which example to run
    # Uncomment the one you want to use

    # Example 1: Basic prediction
    # predictor = example_basic_prediction()

    # Example 2: Classification task
    # predictor = example_classification()

    # Example 3: Custom model parameters
    # predictor = example_custom_model_params()

    # Example 4: Use all features
    # predictor = example_without_feature_selection()

    # Example 5: Cross-validation only
    # predictor = example_cv_only()

    # Example 6: Complete pipeline
    selector, predictor = example_complete_pipeline()

    print("\nNote: Uncomment one of the examples above to run it.")
    print("Make sure to update file paths to point to your actual data files.")
