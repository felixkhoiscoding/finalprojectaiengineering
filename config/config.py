"""
Configuration file for Time-Series Forecasting Project

This file contains all user-configurable parameters.
Modify these values according to your needs.
"""

# ============================================================
# USER INPUT: Forecast Configuration
# ============================================================

# Number of months to forecast ahead
# Default: 12 months (1 year)
# Range: 1-36 months recommended
FORECAST_HORIZON = 12  # USER INPUT: Modify this value

# Number of months to use as test set
# Default: 12 months
# This will be used for model evaluation
TEST_SIZE = 12  # USER INPUT: Modify this value

# ============================================================
# USER INPUT: Model Configuration
# ============================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# LSTM Configuration (if using deep learning)
LSTM_SEQUENCE_LENGTH = 12  # Number of past months to use for prediction
LSTM_UNITS = 64  # Number of LSTM units
LSTM_DROPOUT = 0.2  # Dropout rate for regularization
LSTM_EPOCHS = 100  # Number of training epochs
LSTM_BATCH_SIZE = 16  # Batch size for training

# ============================================================
# File Paths
# ============================================================

import os
from pathlib import Path

# Project root directory
# Project root directory
# Dynamic path for compatibility with Streamlit Cloud
# Method 1: Relative to this config file
PROJECT_ROOT = Path(__file__).parent.parent

# Method 2: Current working directory (backup)
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = Path.cwd()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WPU101704.xlsx"

# Verify raw data path exists, if not try to find it
if not RAW_DATA_PATH.exists():
    # Try looking in current directory
    potential_path = Path("data/raw/WPU101704.xlsx")
    if potential_path.exists():
        RAW_DATA_PATH = potential_path.resolve()
        PROJECT_ROOT = RAW_DATA_PATH.parent.parent.parent
    else:
        # Try looking one level up
        potential_path = Path("../data/raw/WPU101704.xlsx")
        if potential_path.exists():
            RAW_DATA_PATH = potential_path.resolve()

PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FORECASTS_DIR = RESULTS_DIR / "forecasts"

# Notebook paths
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
