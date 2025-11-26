"""
Data Loading Module

This module contains functions for loading and validating time-series data.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import RAW_DATA_PATH


def load_data(filepath=None, sheet_name='Monthly'):
    """
    Load time-series data from Excel file
    
    Parameters:
    -----------
    filepath : str or Path, optional
        Path to the Excel file. If None, uses default path from config
    sheet_name : str, default='Monthly'
        Name of the sheet to read
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with datetime index and target column
        
    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the data format is invalid
    """
    # Use default path if not provided
    if filepath is None:
        filepath = RAW_DATA_PATH
    
    # Check if file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from: {filepath}")
    print(f"Sheet name: {sheet_name}")
    
    # Load data
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    
    # Validate data
    print(f"\nData loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def validate_data(df):
    """
    Validate the loaded data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }
    
    # Check for required columns
    if 'observation_date' not in df.columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Missing 'observation_date' column")
    
    if 'WPU101704' not in df.columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Missing 'WPU101704' column")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        validation_results['warnings'].append(f"Found {missing_count} missing values")
    
    # Check data types
    if 'observation_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['observation_date']):
            validation_results['warnings'].append("'observation_date' is not datetime type")
    
    if 'WPU101704' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['WPU101704']):
            validation_results['is_valid'] = False
            validation_results['issues'].append("'WPU101704' is not numeric type")
    
    # Print validation results
    print("\n" + "="*60)
    print("DATA VALIDATION RESULTS")
    print("="*60)
    
    if validation_results['is_valid']:
        print("[OK] Data is valid!")
    else:
        print("[ERROR] Data validation failed!")
        print("\nIssues found:")
        for issue in validation_results['issues']:
            print(f"  - {issue}")
    
    if validation_results['warnings']:
        print("\nWarnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    print("="*60 + "\n")
    
    return validation_results


def get_data_summary(df):
    """
    Get summary statistics of the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to summarize
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'total_records': len(df),
        'date_range': None,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'basic_stats': None
    }
    
    if 'observation_date' in df.columns:
        summary['date_range'] = {
            'start': df['observation_date'].min(),
            'end': df['observation_date'].max()
        }
    
    if 'WPU101704' in df.columns:
        summary['basic_stats'] = df['WPU101704'].describe().to_dict()
    
    return summary


def print_data_info(df):
    """
    Print comprehensive information about the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    # Basic info
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    if 'observation_date' in df.columns:
        print("\nDate Range:")
        print(f"  Start: {df['observation_date'].min()}")
        print(f"  End: {df['observation_date'].max()}")
        print(f"  Duration: {(df['observation_date'].max() - df['observation_date'].min()).days} days")
    
    # Basic statistics
    if 'WPU101704' in df.columns:
        print("\nTarget Variable Statistics (WPU101704):")
        stats = df['WPU101704'].describe()
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.2f}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader module...\n")
    
    # Load data
    df = load_data()
    
    # Validate data
    validation = validate_data(df)
    
    # Print data info
    print_data_info(df)
    
    # Get summary
    summary = get_data_summary(df)
    print("Summary dictionary created successfully!")
