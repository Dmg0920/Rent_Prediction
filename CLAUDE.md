# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rent prediction ML project for Taipei and New Taipei City properties. The architecture separates ML development (`ml/`) from web deployment (`webapp/`) with shared feature engineering logic (`shared/`) to ensure training/inference consistency.

## Common Commands

All scripts must be run from the project root directory.

```bash
# Data processing pipeline (must run before training)
python ml/scripts/data_pipeline.py

# Train and evaluate models
python ml/src/models/train_model.py

# Build production model (uses 100% data)
python ml/src/models/build_production_model.py

# Advanced statistical analysis
python ml/scripts/advanced_analysis.py

# Run Django web app
cd webapp && python manage.py runserver
```

## Architecture

```
ML Development (ml/)     →    Shared (shared/)    ←    Web Deployment (webapp/)
                              FeatureEngineer
```

**Key Design Principle**: Feature engineering logic lives ONLY in `shared/feature_engineering.py`. Both ML training and web prediction import from this single source to guarantee consistency.

### Data Flow
1. Raw CSV data → DataLoader (adds 坪數, 每坪租金)
2. → DataCleaner (removes non-residential, outliers via IQR)
3. → FeatureEngineer (屋齡, 樓層, building type encoding)
4. → Model training (log-space transformation on price)
5. → Production model (.pkl with scaler and feature metadata)
6. → Django prediction endpoint

### ML Module Structure (`ml/src/`)
- `preprocessing/`: DataLoader, DataCleaner, FeatureEngineer (imports from shared), Visualizer
- `models/`: train_model.py, build_production_model.py, ensemble_models.py, model_utils.py
- `analysis/`: feature_analyzer.py, model_evaluator.py (statistical testing)

### Model Artifacts
Models are saved via joblib with three components: the model object, scaler, and feature list. The web app loads these and reindexes input features to match training exactly.

## Tech Stack
- **ML**: scikit-learn (Ridge, RandomForest, GradientBoosting), XGBoost, LightGBM
- **Statistics**: statsmodels for hypothesis testing
- **Web**: Django 4.2, SQLite
- **Serialization**: joblib

## Important Notes

- All scripts assume execution from project root (not from subdirectories)
- Models train on `log(price)` for better distribution fit
- After modifying `shared/feature_engineering.py`, you must retrain the model
- The web app performs feature alignment by reindexing to match training features exactly
