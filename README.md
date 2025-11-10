# Refined Gradient Boosting

A sophisticated three-stage feature selection method based on LightGBM, designed for high-dimensional biological data with integrated MSA preprocessing capabilities.

## Overview

This repository implements a comprehensive feature selection pipeline:

1. **MSA Preprocessing**: Converts Multiple Sequence Alignment data to optimized one-hot encoded features
2. **Feature Pooling**: Initial feature selection and accumulation through iterative training
3. **Feature Distillation**: Refines the feature pool by removing less important features
4. **Feature Ranking**: Final ranking and selection of the most important features

## Installation

### Prerequisites

- Python 3.7+
- Required packages (see requirements.txt)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. MSA Data Preprocessing
The MSA preprocessor converts multiple sequence alignment data to optimized one-hot encoded features with intelligent dimensionality reduction.

#### Usage
```bash
python msa_preprocessor.py \
  --msa_file <path_to_msa_csv> \
  --output_prefix <output_files_prefix> \
  --min_variant_freq 0.05 \
  --conserved_threshold 0.95 \
  --correlation_threshold 0.95
```

#### Parameters
- `--msa_file`: Path to MSA CSV file (required)

- `--output_prefix`: Prefix for output files (default: "msa_processed")

- `--min_variant_freq`: Minimum frequency for variants (default: 0.05)

- `--conserved_threshold`: Conservation threshold for filtering (default: 0.95)

- `--correlation_threshold`: Correlation threshold for merging variants (default: 0.95)

#### Output Files
- `{prefix}_features.csv`: Processed feature matrix

- `{prefix}_merge_summary.csv`: Summary of all merge operations

- `{prefix}_merge_dict.json`: Detailed dictionary of merge operations

### 2. Refined Gradient Boosting Feature Selection
The main feature selection method with three stages of refinement.

#### Basic Usage

```bash
python refined_gradient_boosting.py \
  --X_path <path_to_features> \
  --y_path <path_to_target> \
  --config config.json \
  --n_features 50 \
  --output_dir ./results
```

#### Advanced Usage with Taxonomy Filtering

```bash
python refined_gradient_boosting.py \
  --X_path processed_features.csv \
  --y_path target_variable.csv \
  --training_samples training_indices.csv \
  --taxonomy_path taxonomy_data.csv \
  --taxonomy_filter "order:Passeriformes" \
  --stage all \
  --n_features 50 \
  --output_dir ./feature_selection_results
```

#### Parameters
- `--X_path`: Path to feature data (CSV or Parquet) - required

- `--y_path`: Path to target variable data - required

- `--config`: Path to configuration JSON file (default: "config.json")

- `--training_samples`: Path to training samples file

- `--output_dir`: Output directory for results (default: "./results")

- `--stage`: Which stage to run: pooling, distillation, ranking, or all (default: "all")

- `--n_features`: Number of final features to select (default: 50)

- `--taxonomy_path`: Path to taxonomy data for filtering

- `--taxonomy_filter`: Taxonomy filter in format "column:value"

#### Output Files
- `feature_pool.csv`: Features from pooling stage

- `feature_distilled.csv`: Features from distillation stage

- `feature_ranking_{n}.csv`: Final ranked features

- `pooling_accuracy.csv`: Accuracy metrics for pooling

- `distillation_accuracy.csv`: Accuracy metrics for distillation

- `ranking_accuracy.csv`: Accuracy metrics for ranking

### File Formats

#### MSA Input Format
```text
sample_id,gene1,gene2,gene3
sample1,ATGCTA,GGCTAA,CCGATT
sample2,ATGCTC,GGCTAA,CCGATT
sample3,ATGCTA,GGCTCA,CCGATT
```

#### Target Variable Format
```text
sample_id,residuals
sample1,0.123
sample2,-0.045
sample3,0.267
```

#### Training Samples Format
```text
sample_id,training_feature_0,training_feature_1
sample1,1,1
sample2,1,0
sample3,0,1
sample4,0,0
```

