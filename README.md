# Steel Price Forecasting - Time Series Analysis

**RuangGuru AI Engineering Bootcamp - Final Project**  
**Developer:** Felix Kho | **GitHub:** felixkhoiscoding

## 📋 Project Overview

This project forecasts Producer Price Index (PPI) for Hot Rolled Steel using multiple time-series models, providing construction contractors with data-driven tools for budget planning and risk management.

**Live Dashboard:** `streamlit run streamlit_app.py` or access [streamli](https://constructionsteelpriceforecast.streamlit.app/)

---

## 🎯 Key Features

- **9 Forecasting Models:** Baseline to Deep Learning (Naive, ARIMA, Prophet, LSTM, etc.)
- **Scenario Analysis:** Multi-model risk assessment (Pessimistic/Expected/Optimistic)
- **Pre-Computed Forecasts:** Instant loading (<1 second) via GPU-generated CSVs
- **Interactive Dashboard:** Professional Streamlit interface with visualizations
- **Historical Fit Analysis:** Visual comparison of model performance on past data

---

## 📊 Dataset

- **Source:** U.S. Bureau of Labor Statistics (FRED)
- **Series:** WPU101704 - Producer Price Index for Hot Rolled Steel
- **Period:** June 1982 - September 2025 (520 months)
- **Frequency:** Monthly
- **Base Index:** June 1982 = 100

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/felixkhoiscoding/[repo-name]
cd Final\ Project

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### Generate New Forecasts (Optional)

```bash
# Pre-compute all forecasts (requires GPU for LSTM)
python generate_forecasts.py
```

---

## 📁 Project Structure

```
Final Project/
├── streamlit_app.py          # Main Streamlit dashboard
├── generate_forecasts.py     # Forecast generation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── config/                    # Configuration files
│   └── config.py             # Paths, test size, parameters
│
├── src/                       # Source code modules
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessing.py      # Data preprocessing
│   ├── baseline_models.py    # Naive, MA, Exponential Smoothing
│   ├── advanced_models.py    # ARIMA, Prophet, LSTM
│   ├── evaluation.py         # Model evaluation metrics
│   └── visualization.py      # Plotting functions
│
├── data/                      # Data directory
│   ├── raw/                  # Original Excel data
│   ├── processed/            # Preprocessed data (train/test splits, LSTM arrays)
│   └── forecasts/            # Pre-computed forecasts (CSV files)
│       ├── arima_forecasts.csv
│       ├── naive_forecasts.csv
│       ├── prophet_forecasts.csv
│       ├── lstm_forecasts.csv
│       ├── *_fitted.csv      # Historical model fits
│       └── forecast_metadata.json
│
├── models/                    # Saved models (if any)
│   └── saved_models/
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   └── 03_preprocessing.ipynb
│
└── results/                   # Analysis results
    ├── all_models_comparison.csv
    ├── figures/              # Generated plots
    └── forecasts/            # Forecast outputs
```

---

## 🤖 Models Implemented

| Model | Category | RMSE (Test) | MAPE (Test) | Forecast Horizon |
|-------|----------|-------------|-------------|------------------|
| **LSTM** | Deep Learning | **6.19** | **1.73%** | 1-12 months |
| **Naive** | Baseline | 8.93 | 3.28% | Any |
| **ARIMA** | Statistical | 12.49 | 4.58% | Any |
| **Prophet** | ML-Based | 36.60 | 13.43% | Any |
| SES | Baseline | 19.33 | 7.10% | Any |
| Holt's | Baseline | 14.70 | 5.39% | Any |
| Holt-Winters | Baseline | 13.97 | 5.13% | Any |
| MA-12 | Baseline | 19.66 | 7.22% | Any |
| Seasonal Naive | Baseline | 20.23 | 7.43% | Any |

**Test Period:** October 2024 - September 2025 (12 months)

---

## 💡 Why Pre-Computed Forecasts?

To enable instant loading and cloud deployment:

1. **Speed:** <1 second vs 30-90 seconds real-time training
2. **Cloud-Ready:** No GPU required on Streamlit Cloud
3. **Consistent:** Reproducible results with random seed
4. **Professional:** Industry standard for production systems

**How it works:**
- Forecasts generated locally using `generate_forecasts.py`
- Saved as CSV files in `data/forecasts/`
- Streamlit app loads from CSV instantly
- Update forecasts monthly when new data available

---

## 📈 Dashboard Features

1. **About This Project** - Background and methodology
2. **Overview** - Dataset statistics and historical trends
3. **Model Comparison** - Performance metrics visualization
4. **Forecast** - Two modes:
   - **Scenario Analysis:** Multi-model risk planning
   - **Single Model:** Individual model forecasts
5. **Documentation** - Technical details and model selection guide

---

## 🎓 For Bootcamp Instructors

**Key Technical Highlights:**
- Proper train/test split (456 train, 64 test)
- 9 models evaluated with multiple metrics
- LSTM with reproducible random seed (seed=42)
- Pre-computation approach for deployment
- Professional error handling and fallback logic

**Honest Disclosures:**
- Pre-computed forecasts acknowledged in UI
- LSTM limited to 12 months (documented)
- Model performance clearly stated (not cherry-picked)
- Academic integrity maintained throughout

---

## 🔧 Technical Requirements

```
python>=3.8
streamlit>=1.45.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
prophet>=1.1.0
tensorflow>=2.13.0
plotly>=5.14.0
openpyxl>=3.1.0
joblib>=1.2.0
```

---

## 📝 License

This project is created for educational purposes as part of RuangGuru AI Engineering Bootcamp Final Project.

---

## 📧 Contact

**Felix Kho**  
GitHub: [@felixkhoiscoding](https://github.com/felixkhoiscoding)

---

## 🙏 Acknowledgments

- **Data Source:** U.S. Bureau of Labor Statistics (FRED)
- **Bootcamp:** RuangGuru AI Engineering Program
- **Frameworks:** Streamlit, TensorFlow, Prophet, Statsmodels
