"""
Refined Gradient Boosting - A LightGBM-based feature selection method

This module implements a three-stage feature selection method:
1. Feature Pooling: Initial feature selection and accumulation
2. Feature Distillation: Refining the feature pool
3. Feature Ranking: Final ranking and selection of features

Author: Your Name
License: MIT
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RefinedGradientBoosting')


class Config:
    """Configuration class for Refined Gradient Boosting"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration from JSON file or use defaults
        
        Args:
            config_path: Path to JSON configuration file
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            self._config = self._get_default_config()
            if config_path:
                logger.warning(f"Config file {config_path} not found, using defaults")
            else:
                logger.info("No config file provided, using defaults")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model_params": {
                "learning_rate": 0.05,
                "n_estimators": 512,
                "num_leaves": 32,
                "min_child_samples": 2,
                "early_stopping_round": 64,
                "n_jobs": 4,
                "num_threads": 4,
                "random_state": 42,
                "importance_type": "shap"
            },
            "feature_pooling": {
                "step": 80,
                "n_iter": 10,
                "refinement_iter": 100,
                "max_features": 10000,
                "correlation_threshold": 0.05
            },
            "feature_distillation": {
                "step": 100,
                "top_n": 30,
                "n_iter": 100,
                "refinement_iter": 100,
                "max_attempts": 5,
                "max_features": 1000
            },
            "feature_ranking": {
                "n_iter": 200
            },
            "data_settings": {
                "test_size": 0.1,
                "validation_size": 0.1
            }
        }
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value by section and key"""
        try:
            return self._config[section][key]
        except KeyError:
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"Configuration saved to {path}")


class RefinedGradientBoosting:
    """
    Refined Gradient Boosting feature selection method
    
    This class implements a three-stage feature selection approach
    using LightGBM and SHAP values for robust feature importance estimation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Refined Gradient Boosting feature selector
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_pool = []
        self.feature_importance_history = []
        self.performance_history = []
        
        # Set random seed for reproducibility
        random_state = config.get('model_params', 'random_state')
        if random_state:
            np.random.seed(random_state)
    
    def load_data(self, X_path: str, y_path: str, 
                  training_samples_path: Optional[str] = None,
                  taxonomy_filter: Optional[Dict] = None) -> None:
        """
        Load and prepare data for feature selection
        
        Args:
            X_path: Path to feature data (parquet or CSV)
            y_path: Path to target variable data
            training_samples_path: Optional path to training samples
            taxonomy_filter: Optional dictionary for filtering taxonomy
        """
        logger.info("Loading data...")
        
        # Load feature data
        if X_path.endswith('.parquet'):
            self.X_all = pd.read_parquet(X_path)
        else:
            self.X_all = pd.read_csv(X_path, index_col=0)
        
        # Load target data
        self.y_all = pd.read_csv(y_path, index_col=0)
        if 'residuals' in self.y_all.columns:
            self.y_all = self.y_all['residuals']
        
        # Apply taxonomy filter if provided
        if taxonomy_filter:
            taxonomy_df = pd.read_csv(taxonomy_filter['path'])
            for col, value in taxonomy_filter['filters'].items():
                taxonomy_df = taxonomy_df.loc[taxonomy_df[col] == value]
            self.X_all = self.X_all.loc[taxonomy_df[taxonomy_filter['index_col']]]
            self.y_all = self.y_all.loc[taxonomy_df[taxonomy_filter['index_col']]]
        
        # Ensure indices match
        if not all(self.X_all.index == self.y_all.index):
            logger.warning("X and y indices don't match perfectly, aligning...")
            common_idx = self.X_all.index.intersection(self.y_all.index)
            self.X_all = self.X_all.loc[common_idx]
            self.y_all = self.y_all.loc[common_idx]
        
        # Split data
        if training_samples_path and os.path.exists(training_samples_path):
            training_samples = pd.read_csv(training_samples_path, index_col=0)
            training_indices = list(training_samples['training_feature'])
        else:
            # Default split if no training samples provided
            training_indices = self._create_default_split()
        
        self.X_train = self.X_all.loc[training_indices]
        self.y_train = self.y_all.loc[training_indices]
        self.X_test = self.X_all.loc[~self.X_all.index.isin(training_indices)]
        self.y_test = self.y_all.loc[~self.X_all.index.isin(training_indices)]
        
        logger.info(f"Data loaded: {self.X_train.shape[0]} training, {self.X_test.shape[0]} test samples")
        logger.info(f"Features: {self.X_train.shape[1]}")
    
    def _create_default_split(self) -> List:
        """Create default train/test split"""
        test_size = self.config.get('data_settings', 'test_size', 0.2)
        random_state = self.config.get('model_params', 'random_state', 42)
        
        _, training_indices = train_test_split(
            self.X_all.index, 
            test_size=test_size, 
            random_state=random_state
        )
        return list(training_indices)
    
    def _create_model(self, params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """Create LightGBM model with specified parameters"""
        # Get base model parameters from config
        model_config = self.config.get_section('model_params')
        
        default_params = {
            'device': 'cpu',
            'n_jobs': model_config.get('n_jobs', 1),
            'num_threads': model_config.get('num_threads', 25),
            'n_estimators': model_config.get('n_estimators', 512),
            'num_leaves': model_config.get('num_leaves', 32),
            'min_child_samples': model_config.get('min_child_samples', 2),
            'learning_rate': model_config.get('learning_rate', 0.05),
            'verbose': -1,
            'random_state': model_config.get('random_state', 42)
        }
        
        if params:
            default_params.update(params)
        
        return lgb.LGBMRegressor(**default_params)
    
    def _compute_feature_importance(self, model, X: pd.DataFrame, 
                                  cor_test: float, 
                                  importance_type: str = 'shap') -> pd.DataFrame:
        """Compute feature importance using SHAP or built-in importance"""
        feature_names = X.columns
        
        if importance_type == 'shap':
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            vals = np.abs(shap_values.values).mean(0)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': cor_test * vals
            })
        else:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': cor_test * model.feature_importances_
            })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on multiple datasets"""
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        results = {
            'cor_train': pd.Series(y_train).corr(pd.Series(y_pred_train)),
            'cor_val': pd.Series(y_val).corr(pd.Series(y_pred_val)),
            'cor_test': pd.Series(y_test).corr(pd.Series(y_pred_test)),
            'mse_val': mean_squared_error(y_val, y_pred_val),
            'mae_val': mean_absolute_error(y_val, y_pred_val)
        }
        
        return results
    
    def _train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          n_estimators_multiplier: int = 1) -> Tuple[lgb.LGBMRegressor, Dict]:
        """Train model and evaluate performance"""
        # Split training data
        validation_size = self.config.get('data_settings', 'validation_size', 0.1)
        random_state = self.config.get('model_params', 'random_state', 42)
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, 
            random_state=random_state
        )
        
        # Create and train model
        model_params = {}
        if n_estimators_multiplier > 1:
            base_estimators = self.config.get('model_params', 'n_estimators', 512)
            model_params['n_estimators'] = base_estimators * n_estimators_multiplier
        
        model = self._create_model(model_params)
        
        # Get early stopping rounds
        early_stopping_rounds = self.config.get('model_params', 'early_stopping_round', 64)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val), (X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # Evaluate
        results = self._evaluate_model(model, X_tr, y_tr, X_val, y_val, X_test, y_test)
        
        return model, results
    
    def feature_pooling(self, output_dir: str = './results') -> List[str]:
        """
        Stage 1: Feature Pooling
        
        Accumulate features through iterative training and importance calculation
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            List of selected features
        """
        logger.info("Starting Feature Pooling stage...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        feature_dir = os.path.join(output_dir, 'feature_pool.csv')
        raw_feature_dir = os.path.join(output_dir, 'feature_pool_raw.csv')
        accuracy_dir = os.path.join(output_dir, 'pooling_accuracy.csv')
        
        # Initialize output files
        with open(accuracy_dir, 'w') as f:
            f.write('n_features,cor_val,cor_test,ave_cor_val,ave_cor_test\n')
        
        with open(raw_feature_dir, 'w') as f:
            f.write('iteration,value,feature\n')
        
        pooling_config = self.config.get_section('feature_pooling')
        step = pooling_config.get('step', 80)
        thres = pooling_config.get('correlation_threshold', 0.05)
        max_features = pooling_config.get('max_features', 10000)
        importance_type = self.config.get('model_params', 'importance_type', 'shap')
        
        i = 0
        while len(self.feature_pool) < max_features:
            logger.info(f"Iteration {i}: Current feature pool size: {len(self.feature_pool)}")
            
            summary_importance, accuracy_metrics = self._run_pooling_iteration(i, pooling_config, importance_type)
            
            if summary_importance is None:
                logger.warning(f"No valid models in iteration {i}, stopping.")
                break
            
            # Select top features
            feature_imp = summary_importance.mean(axis=1).sort_values(ascending=False)
            top_features = feature_imp.head(step).index.tolist()
            
            # Add new features to pool
            new_features = [f for f in top_features if f not in self.feature_pool]
            self.feature_pool.extend(new_features)
            
            # Save raw feature importance
            with open(raw_feature_dir, 'a') as f:
                for feature, importance in feature_imp.head(step).items():
                    f.write(f'{i},{importance},{feature}\n')
            
            # Refine feature pool
            self._refine_feature_pool(accuracy_dir, i, pooling_config)
            
            # Save current feature pool
            pd.DataFrame({'feature': self.feature_pool}).to_csv(feature_dir)
            
            if not new_features:
                logger.info("No new features added, stopping feature pooling.")
                break
                
            i += 1
        
        logger.info(f"Feature pooling completed. Final pool size: {len(self.feature_pool)}")
        return self.feature_pool
    
    def _run_pooling_iteration(self, iteration: int, pooling_config: Dict, 
                             importance_type: str) -> Tuple[Optional[pd.DataFrame], List]:
        """Run a single iteration of feature pooling"""
        summary_importance = pd.DataFrame(index=self.X_train.columns)
        accuracy_metrics = []
        
        n_iter = pooling_config.get('n_iter', 10)
        correlation_threshold = pooling_config.get('correlation_threshold', 0.05)
        
        j = 0
        while j < n_iter:
            # Create feature subset
            X_sub = self._create_feature_subset()
            
            # Train and evaluate model
            model, results = self._train_and_evaluate(
                X_sub, self.y_train, self.X_test[X_sub.columns], self.y_test
            )
            
            if results['cor_val'] > correlation_threshold:
                # Compute feature importance
                importance_df = self._compute_feature_importance(
                    model, X_sub, results['cor_val'], importance_type
                )
                
                # Add to summary
                col_name = f'shap_vals_{j}'
                temp_df = importance_df.set_index('feature')['importance'].rename(col_name)
                summary_importance = pd.concat([summary_importance, temp_df], axis=1)
                
                accuracy_metrics.append(results)
                logger.info(f'{iteration}-{j}: Train={results["cor_train"]:.3f}, '
                          f'Val={results["cor_val"]:.3f}, Test={results["cor_test"]:.3f}')
                j += 1
        
        if summary_importance.empty:
            return None, []
            
        return summary_importance, accuracy_metrics
    
    def _create_feature_subset(self) -> pd.DataFrame:
        """Create a subset of features for training"""
        if not self.feature_pool:
            return self.X_train.copy()
        
        # Drop some features from pool with probability based on position
        dropped_col = []
        if len(self.feature_pool) > 30:
            dropped_col = self.feature_pool[30:]
        
        return self.X_train.drop(dropped_col, axis=1)
    
    def _refine_feature_pool(self, accuracy_dir: str, iteration: int, pooling_config: Dict) -> None:
        """Refine the current feature pool with more extensive training"""
        if not self.feature_pool:
            return
        
        X_sub = self.X_train[self.feature_pool]
        summary_importance = pd.DataFrame(index=X_sub.columns)
        accuracy_metrics = []
        
        refinement_iter = pooling_config.get('refinement_iter', 100)
        
        for q in range(refinement_iter):
            model, results = self._train_and_evaluate(
                X_sub, self.y_train, self.X_test[self.feature_pool], self.y_test,
                n_estimators_multiplier=3  # Use more trees for refinement
            )
            
            # Compute SHAP importance
            importance_df = self._compute_feature_importance(
                model, X_sub, results['cor_val'], 'shap'
            )
            
            # Add to summary
            col_name = f'refine_vals_{q}'
            temp_df = importance_df.set_index('feature')['importance'].rename(col_name)
            summary_importance = pd.concat([summary_importance, temp_df], axis=1)
            
            accuracy_metrics.append(results)
            logger.info(f'Refine {iteration}-{q}: Train={results["cor_train"]:.3f}, '
                      f'Val={results["cor_val"]:.3f}, Test={results["cor_test"]:.3f}')
        
        # Update feature pool based on refined importance
        feature_imp = summary_importance.mean(axis=1).sort_values(ascending=False)
        self.feature_pool = feature_imp.index.tolist()
        
        # Save accuracy metrics
        cor_val_mean = np.mean([r['cor_val'] for r in accuracy_metrics])
        cor_test_mean = np.mean([r['cor_test'] for r in accuracy_metrics])
        
        with open(accuracy_dir, 'a') as f:
            f.write(f'{len(self.feature_pool)},{cor_val_mean:.4f},{cor_test_mean:.4f},'
                   f'{cor_val_mean:.4f},{cor_test_mean:.4f}\n')
    
    def feature_distillation(self, initial_features: List[str], 
                           output_dir: str = './results') -> List[str]:
        """
        Stage 2: Feature Distillation
        
        Refine the feature pool by removing less important features
        
        Args:
            initial_features: List of features from pooling stage
            output_dir: Directory to save results
            
        Returns:
            List of distilled features
        """
        logger.info("Starting Feature Distillation stage...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        feature_dir = os.path.join(output_dir, 'feature_distilled.csv')
        accuracy_dir = os.path.join(output_dir, 'distillation_accuracy.csv')
        
        with open(accuracy_dir, 'w') as f:
            f.write('n_features,cor_val,cor_test,ave_cor_val,ave_cor_test\n')
        
        distillation_config = self.config.get_section('feature_distillation')
        step = distillation_config.get('step', 100)
        top_n = distillation_config.get('top_n', 30)
        max_attempts = distillation_config.get('max_attempts', 5)
        max_features = distillation_config.get('max_features', 1000)
        importance_type = self.config.get('model_params', 'importance_type', 'shap')
        
        feature_pool = initial_features.copy()
        current_attempt = 0
        i = 0
        
        while len(feature_pool) < max_features:
            logger.info(f"Distillation iteration {i}: {len(feature_pool)} features")
            
            summary_importance, accuracy_metrics = self._run_distillation_iteration(
                feature_pool, distillation_config, importance_type
            )
            
            if summary_importance.empty:
                logger.warning(f"No valid models in distillation iteration {i}")
                break
            
            # Select top features
            feature_imp = summary_importance.mean(axis=1).sort_values(ascending=False)
            top_features = feature_imp.head(step).index.tolist()
            
            # Add new features
            new_features = [f for f in top_features if f not in feature_pool]
            feature_pool.extend(new_features)
            
            # Check for convergence
            if not new_features:
                current_attempt += 1
                if current_attempt >= max_attempts:
                    logger.info("Max attempts reached without new features, stopping distillation.")
                    break
            else:
                current_attempt = 0
            
            # Refine and save
            feature_pool = self._refine_distillation_pool(
                feature_pool, accuracy_dir, i, distillation_config
            )
            pd.DataFrame({'feature': feature_pool}).to_csv(feature_dir)
            
            i += 1
        
        logger.info(f"Feature distillation completed. Final features: {len(feature_pool)}")
        return feature_pool
    
    def _run_distillation_iteration(self, feature_pool: List[str], 
                                  distillation_config: Dict,
                                  importance_type: str) -> Tuple[pd.DataFrame, List]:
        """Run a single iteration of feature distillation"""
        summary_importance = pd.DataFrame(index=self.X_train.columns)
        accuracy_metrics = []
        
        n_iter = distillation_config.get('n_iter', 100)
        correlation_threshold = distillation_config.get('correlation_threshold', 0.05)
        
        j = 0
        while j < n_iter:
            # Create feature subset
            if feature_pool:
                dropped_col = feature_pool[distillation_config.get('top_n', 30):]
                X_sub = self.X_train.drop(dropped_col, axis=1)
            else:
                X_sub = self.X_train.copy()
            
            # Train and evaluate
            model, results = self._train_and_evaluate(
                X_sub, self.y_train, self.X_test[X_sub.columns], self.y_test
            )
            
            if results['cor_val'] > correlation_threshold:
                importance_df = self._compute_feature_importance(
                    model, X_sub, results['cor_val'], importance_type
                )
                
                col_name = f'distill_vals_{j}'
                temp_df = importance_df.set_index('feature')['importance'].rename(col_name)
                summary_importance = pd.concat([summary_importance, temp_df], axis=1)
                
                accuracy_metrics.append(results)
                logger.info(f'Distill {j}: Val={results["cor_val"]:.3f}, Test={results["cor_test"]:.3f}')
                j += 1
        
        return summary_importance, accuracy_metrics
    
    def _refine_distillation_pool(self, feature_pool: List[str], 
                                accuracy_dir: str, iteration: int,
                                distillation_config: Dict) -> List[str]:
        """Refine the distillation feature pool"""
        X_sub = self.X_train[feature_pool]
        summary_importance = pd.DataFrame(index=X_sub.columns)
        accuracy_metrics = []
        
        refinement_iter = distillation_config.get('refinement_iter', 100)
        
        for q in range(refinement_iter):
            model, results = self._train_and_evaluate(
                X_sub, self.y_train, self.X_test[feature_pool], self.y_test
            )
            
            importance_df = self._compute_feature_importance(
                model, X_sub, results['cor_val'], 'shap'
            )
            
            col_name = f'distill_refine_{q}'
            temp_df = importance_df.set_index('feature')['importance'].rename(col_name)
            summary_importance = pd.concat([summary_importance, temp_df], axis=1)
            
            accuracy_metrics.append(results)
        
        # Update feature pool based on refined importance
        feature_imp = summary_importance.mean(axis=1).sort_values(ascending=False)
        refined_pool = feature_imp.index.tolist()
        
        # Save accuracy
        cor_val_mean = np.mean([r['cor_val'] for r in accuracy_metrics])
        cor_test_mean = np.mean([r['cor_test'] for r in accuracy_metrics])
        
        with open(accuracy_dir, 'a') as f:
            f.write(f'{len(refined_pool)},{cor_val_mean:.4f},{cor_test_mean:.4f},'
                   f'{cor_val_mean:.4f},{cor_test_mean:.4f}\n')
        
        return refined_pool
    
    def feature_ranking(self, features: List[str], n_features: int,
                      output_dir: str = './results') -> pd.DataFrame:
        """
        Stage 3: Feature Ranking
        
        Final ranking and selection of features
        
        Args:
            features: List of features from distillation stage
            n_features: Number of top features to select
            output_dir: Directory to save results
            
        Returns:
            DataFrame with ranked features and their importance scores
        """
        logger.info("Starting Feature Ranking stage...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        accuracy_dir = os.path.join(output_dir, 'ranking_accuracy.csv')
        feature_dir = os.path.join(output_dir, f'feature_ranking_{n_features}.csv')
        
        with open(accuracy_dir, 'w') as f:
            f.write('n_features,cor_val,cor_test,mse_val,mse_test,mae_val,mae_test\n')
        
        ranking_config = self.config.get_section('feature_ranking')
        
        selected_features = features.copy()
        i = 0
        
        while True:
            logger.info(f"Ranking iteration {i}: {len(selected_features)} features")
            
            # Evaluate current feature set
            ranking_results = self._run_ranking_iteration(
                selected_features, accuracy_dir, i, ranking_config
            )
            
            if ranking_results is None:
                break
            
            feature_imp, performance = ranking_results
            
            # Save current ranking
            feature_imp.to_csv(os.path.join(
                output_dir, f'iter{i}_feature_ranking_{n_features}.csv'
            ))
            
            # Check stopping condition
            if len(selected_features) <= n_features:
                break
            
            # Reduce feature set
            selected_features = self._reduce_feature_set(selected_features, feature_imp)
            
            i += 1
        
        # Save final ranking
        final_ranking = self._get_final_ranking(features, n_features)
        final_ranking.to_csv(feature_dir)
        
        logger.info(f"Feature ranking completed. Top {n_features} features selected.")
        return final_ranking
    
    def _run_ranking_iteration(self, features: List[str], 
                             accuracy_dir: str, iteration: int,
                             ranking_config: Dict) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """Run a single iteration of feature ranking"""
        if not features:
            return None
        
        X_sub = self.X_train[features]
        summary_importance = pd.DataFrame(index=X_sub.columns)
        performance_metrics = []
        
        n_iter = ranking_config.get('n_iter', 200)
        
        for j in range(n_iter):
            model, results = self._train_and_evaluate(
                X_sub, self.y_train, self.X_test[features], self.y_test
            )
            
            # Compute SHAP importance
            importance_df = self._compute_feature_importance(
                model, X_sub, results['cor_val'], 'shap'
            )
            
            col_name = f'ranking_vals_{j}'
            temp_df = importance_df.set_index('feature')['importance'].rename(col_name)
            summary_importance = pd.concat([summary_importance, temp_df], axis=1)
            
            performance_metrics.append(results)
        
        # Calculate average importance and performance
        feature_imp = summary_importance.mean(axis=1).sort_values(ascending=False)
        feature_imp_df = pd.DataFrame({
            'feature': feature_imp.index,
            'importance': feature_imp.values
        })
        
        # Calculate average performance
        avg_performance = {
            'cor_val': np.mean([r['cor_val'] for r in performance_metrics]),
            'cor_test': np.mean([r['cor_test'] for r in performance_metrics]),
            'mse_val': np.mean([r['mse_val'] for r in performance_metrics]),
            'mse_test': np.mean([r['mse_test'] for r in performance_metrics]),
            'mae_val': np.mean([r['mae_val'] for r in performance_metrics]),
            'mae_test': np.mean([r['mae_test'] for r in performance_metrics])
        }
        
        logger.info(f'Ranking {iteration}: {len(features)} features, '
                   f'Val Corr={avg_performance["cor_val"]:.3f}, '
                   f'Test Corr={avg_performance["cor_test"]:.3f}')
        
        # Save performance
        with open(accuracy_dir, 'a') as f:
            f.write(f'{len(features)},{avg_performance["cor_val"]:.4f},'
                   f'{avg_performance["cor_test"]:.4f},{avg_performance["mse_val"]:.4f},'
                   f'{avg_performance["mse_test"]:.4f},{avg_performance["mae_val"]:.4f},'
                   f'{avg_performance["mae_test"]:.4f}\n')
        
        return feature_imp_df, avg_performance
    
    def _reduce_feature_set(self, current_features: List[str], 
                          feature_imp: pd.DataFrame) -> List[str]:
        """Reduce the feature set based on importance"""
        n_current = len(current_features)
        
        if n_current > 1000:
            return feature_imp['feature'].head(1000).tolist()
        elif n_current > 500:
            return feature_imp['feature'].head(n_current - 100).tolist()
        elif n_current > 200:
            return feature_imp['feature'].head(n_current - 50).tolist()
        elif n_current > 100:
            return feature_imp['feature'].head(n_current - 20).tolist()
        else:
            return feature_imp['feature'].head(n_current - 10).tolist()
    
    def _get_final_ranking(self, features: List[str], n_features: int) -> pd.DataFrame:
        """Get final feature ranking"""
        # Use the last ranking iteration results
        ranking_file = f'iter0_feature_ranking_{n_features}.csv'
        if os.path.exists(ranking_file):
            return pd.read_csv(ranking_file, index_col=0)
        else:
            # Fallback: simple ranking based on correlation with target
            correlations = []
            for feature in features:
                corr = self.X_train[feature].corr(self.y_train)
                correlations.append((feature, abs(corr)))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            return pd.DataFrame({
                'feature': [x[0] for x in correlations],
                'importance': [x[1] for x in correlations]
            }).head(n_features)


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Refined Gradient Boosting Feature Selection')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.json', 
                       help='Path to configuration JSON file')
    
    # Data parameters
    parser.add_argument('--X_path', type=str, required=True, help='Path to feature data')
    parser.add_argument('--y_path', type=str, required=True, help='Path to target data')
    parser.add_argument('--training_samples', type=str, help='Path to training samples')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    # Stage selection
    parser.add_argument('--stage', type=str, choices=['pooling', 'distillation', 'ranking', 'all'],
                       default='all', help='Which stage to run')
    
    # Feature selection parameters
    parser.add_argument('--n_features', type=int, default=50, help='Number of final features')
    parser.add_argument('--y_type', type=str, help='Target variable type (for file naming)')
    parser.add_argument('--sample_id', type=str, help='Sample ID (for file naming)')
    
    # Taxonomy filter
    parser.add_argument('--taxonomy_path', type=str, help='Path to taxonomy data')
    parser.add_argument('--taxonomy_filter', type=str, help='Taxonomy filter (e.g., "order:Passeriformes")')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup taxonomy filter if provided
    taxonomy_filter = None
    if args.taxonomy_path and args.taxonomy_filter:
        filter_col, filter_value = args.taxonomy_filter.split(':')
        taxonomy_filter = {
            'path': args.taxonomy_path,
            'filters': {filter_col: filter_value},
            'index_col': 'species_name'
        }
    
    # Initialize feature selector
    selector = RefinedGradientBoosting(config)
    
    # Load data
    selector.load_data(
        X_path=args.X_path,
        y_path=args.y_path,
        training_samples_path=args.training_samples,
        taxonomy_filter=taxonomy_filter
    )
    
    # Run selected stages
    if args.stage in ['pooling', 'all']:
        features_pool = selector.feature_pooling(args.output_dir)
        logger.info(f"Feature pooling completed: {len(features_pool)} features")
    
    if args.stage in ['distillation', 'all']:
        features_distilled = selector.feature_distillation(
            selector.feature_pool, args.output_dir
        )
        logger.info(f"Feature distillation completed: {len(features_distilled)} features")
    
    if args.stage in ['ranking', 'all']:
        features_ranked = selector.feature_ranking(
            selector.feature_pool, args.n_features, args.output_dir
        )
        logger.info(f"Feature ranking completed: {len(features_ranked)} top features")
    
    logger.info("Refined Gradient Boosting feature selection completed!")


if __name__ == '__main__':
    main()