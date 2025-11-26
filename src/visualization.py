"""
Visualization Module

This module contains reusable plotting functions for time-series analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def plot_time_series(data, date_col, value_col, title='Time Series Plot', 
                     figsize=(16, 6), save_path=None):
    """
    Create a clean time series line plot
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the time series data
    date_col : str
        Name of the date column
    value_col : str
        Name of the value column to plot
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(data[date_col], data[value_col], linewidth=1.5, 
            color='darkblue', alpha=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel(value_col, fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig, ax


def plot_decomposition(data, date_col, value_col, model='additive', 
                       period=12, figsize=(16, 10), save_path=None):
    """
    Decompose time series into trend, seasonal, and residual components
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the time series data
    date_col : str
        Name of the date column
    value_col : str
        Name of the value column
    model : str, default='additive'
        Type of decomposition: 'additive' or 'multiplicative'
    period : int, default=12
        Period for seasonal decomposition (12 for monthly data)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    decomposition : statsmodels decomposition object
    """
    # Set date as index
    ts_data = data.set_index(date_col)[value_col]
    
    # Perform decomposition
    decomposition = seasonal_decompose(ts_data, model=model, period=period)
    
    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Original
    axes[0].plot(decomposition.observed, color='darkblue', linewidth=1.5)
    axes[0].set_ylabel('Observed', fontsize=12)
    axes[0].set_title(f'Time Series Decomposition ({model.capitalize()})', 
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(decomposition.trend, color='red', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(decomposition.seasonal, color='green', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(decomposition.resid, color='purple', linewidth=1.5)
    axes[3].set_ylabel('Residual', fontsize=12)
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decomposition plot saved to: {save_path}")
    
    return decomposition


def plot_acf_pacf(data, value_col, lags=40, figsize=(16, 6), save_path=None):
    """
    Plot ACF and PACF for time series data
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    value_col : str or None
        Column name if data is DataFrame, None if Series
    lags : int
        Number of lags to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    # Extract series if DataFrame
    if isinstance(data, pd.DataFrame):
        series = data[value_col]
    else:
        series = data
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF plot
    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Lags', fontsize=11)
    axes[0].set_ylabel('ACF', fontsize=11)
    
    # PACF plot
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05)
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Lags', fontsize=11)
    axes[1].set_ylabel('PACF', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ACF/PACF plot saved to: {save_path}")
    
    return fig, axes


def plot_distribution(data, value_col, figsize=(16, 5), save_path=None):
    """
    Plot distribution analysis: histogram and box plot
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    value_col : str
        Column name to analyze
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with KDE
    axes[0].hist(data[value_col], bins=50, color='skyblue', 
                edgecolor='black', alpha=0.7, density=True)
    
    # Add KDE
    data[value_col].plot(kind='kde', ax=axes[0], color='red', 
                         linewidth=2, label='KDE')
    
    # Add mean and median lines
    mean_val = data[value_col].mean()
    median_val = data[value_col].median()
    
    axes[0].axvline(mean_val, color='darkred', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[0].axvline(median_val, color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {median_val:.2f}')
    
    axes[0].set_title('Distribution of Values', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    box = axes[1].boxplot(data[value_col], vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
    
    axes[1].set_title('Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to: {save_path}")
    
    return fig, axes


def plot_rolling_statistics(data, date_col, value_col, windows=[12, 24], 
                            figsize=(16, 6), save_path=None):
    """
    Plot rolling mean and standard deviation
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the time series
    date_col : str
        Date column name
    value_col : str
        Value column name
    windows : list
        List of window sizes for rolling statistics
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot original series
    axes[0].plot(data[date_col], data[value_col], 
                label='Original', color='blue', alpha=0.6, linewidth=1)
    
    # Plot rolling means
    colors = ['red', 'green', 'orange', 'purple']
    for i, window in enumerate(windows):
        rolling_mean = data[value_col].rolling(window=window).mean()
        axes[0].plot(data[date_col], rolling_mean, 
                    label=f'{window}-Month MA', 
                    color=colors[i % len(colors)], linewidth=2)
    
    axes[0].set_title('Rolling Mean', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot rolling standard deviation
    axes[1].plot(data[date_col], data[value_col].rolling(window=12).std(), 
                label='12-Month Rolling Std', color='red', linewidth=2)
    
    if len(windows) > 1:
        axes[1].plot(data[date_col], data[value_col].rolling(window=windows[1]).std(), 
                    label=f'{windows[1]}-Month Rolling Std', color='green', linewidth=2)
    
    axes[1].set_title('Rolling Standard Deviation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Std Dev', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rolling statistics plot saved to: {save_path}")
    
    return fig, axes


def plot_yearly_comparison(data, date_col, value_col, figsize=(16, 6), save_path=None):
    """
    Plot year-over-year comparison
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with time series data
    date_col : str
        Date column name
    value_col : str
        Value column name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Add year and month columns
    data_copy = data.copy()
    data_copy['year'] = data_copy[date_col].dt.year
    data_copy['month'] = data_copy[date_col].dt.month
    
    # Get last 5 years for clarity
    recent_years = sorted(data_copy['year'].unique())[-5:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for year in recent_years:
        year_data = data_copy[data_copy['year'] == year]
        ax.plot(year_data['month'], year_data[value_col], 
               marker='o', label=str(year), linewidth=2)
    
    ax.set_title('Year-over-Year Comparison (Last 5 Years)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(loc='best', title='Year')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Yearly comparison plot saved to: {save_path}")
    
    return fig, ax


def plot_changes(data, date_col, value_col, figsize=(16, 10), save_path=None):
    """
    Plot month-over-month and year-over-year changes
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with time series data
    date_col : str
        Date column name
    value_col : str
        Value column name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    data_copy = data.copy()
    
    # Calculate changes
    data_copy['MoM_change'] = data_copy[value_col].pct_change() * 100
    data_copy['YoY_change'] = data_copy[value_col].pct_change(periods=12) * 100
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Month-over-Month changes
    axes[0].plot(data_copy[date_col], data_copy['MoM_change'], 
                color='blue', linewidth=1.5, alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_title('Month-over-Month % Change', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('% Change', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Year-over-Year changes
    axes[1].plot(data_copy[date_col], data_copy['YoY_change'], 
                color='green', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('Year-over-Year % Change', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('% Change', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Changes plot saved to: {save_path}")
    
    return fig, axes


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("  - plot_time_series()")
    print("  - plot_decomposition()")
    print("  - plot_acf_pacf()")
    print("  - plot_distribution()")
    print("  - plot_rolling_statistics()")
    print("  - plot_yearly_comparison()")
    print("  - plot_changes()")
