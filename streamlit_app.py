"""
Streamlit App for Time-Series Forecasting
Producer Price Index (Hot Rolled Steel) Forecasting Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from src.baseline_models import naive_forecast, holt_winters
from src.advanced_models import train_prophet
from config.config import RAW_DATA_PATH, TEST_SIZE, FORECAST_HORIZON

# Page config
st.set_page_config(
    page_title="Hot Rolled Steel Price Forecasting",
    page_icon="🏗️",
    layout="wide"
)
#Caption


# Title and Caption 
st.title("🏗️ Hot Rolled Steel Price Forecasting")
st.markdown("**Time Series Analysis & Risk Management**")
st.caption("""Made by : Felix Kho  
           Github : felixkhoiscoding    
           Data source : [FRED - WPU101704](https://fred.stlouisfed.org/series/WPU101704)""")

#Data last updated information
st.markdown("**Data Last Updated:** 30 September 2025")


# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🎯 About This Project", "📈 Overview", "🔍 Model Comparison", "🔮 Forecast", "📚 Documentation"])

# Load data
@st.cache_data
def load_project_data():
    if not RAW_DATA_PATH.exists():
        st.error(f"❌ File not found at: {RAW_DATA_PATH}")
        # List contents of parent directories to debug
        if RAW_DATA_PATH.parent.exists():
            st.write(f"Contents of {RAW_DATA_PATH.parent}:")
            st.write([p.name for p in RAW_DATA_PATH.parent.iterdir()])
        else:
            st.error(f"Directory {RAW_DATA_PATH.parent} does not exist!")
            if RAW_DATA_PATH.parent.parent.exists():
                st.write(f"Contents of {RAW_DATA_PATH.parent.parent}:")
                st.write([p.name for p in RAW_DATA_PATH.parent.parent.iterdir()])
    
    df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')
    split_idx = len(df) - TEST_SIZE
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return df, train, test

# Load pre-computed forecasts
@st.cache_data
def load_precomputed_forecast(model_name, horizon_months):
    """Load pre-computed forecast and fitted values from CSV files"""
    try:
        forecasts_dir = Path("data/forecasts")
        
        # Load forecast for specific horizon
        forecast_file = forecasts_dir / f"{model_name}_forecasts.csv"
        df_forecast = pd.read_csv(forecast_file)
        df_horizon = df_forecast[df_forecast['horizon_months'] == horizon_months].copy()
        
        # Load fitted values
        fitted_file = forecasts_dir / f"{model_name}_fitted.csv"
        df_fitted = pd.read_csv(fitted_file)
        
        if len(df_horizon) == 0:
            return None, None
            
        # Extract forecast values
        forecast_values = df_horizon.sort_values('step')['forecast'].values
        fitted_values = df_fitted['fitted'].values
        
        return forecast_values, fitted_values
        
    except (FileNotFoundError, KeyError) as e:
        st.warning(f"Pre-computed {model_name} forecasts not found. Falling back to real-time generation.")
        return None, None

df, train, test = load_project_data()

# PAGE 0: About This Project
if page == "🎯 About This Project":
    st.header("About This Project")
    
    # Project Badge
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
            <h2>🎓 RuangGuru AI Engineering Bootcamp</h2>
            <h3>Final Project - Time Series Forecasting</h3>
            <p><strong>By:</strong> Felix Kho</p>
            <p><strong>GitHub:</strong> felixkhoiscoding | <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/felixkho/" style="color: #e0e0ff;">felixkho</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Problem Statement
    st.subheader("🎯 Problem Statement")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Construction projects face significant financial risks due to material price volatility.** 
        Hot rolled steel, a critical construction material, experiences unpredictable price fluctuations 
        that can derail project budgets and profitability.
        
        **Key Challenges:**
        - 📊 **Price Uncertainty:** Steel prices can swing ±15-35% within 6 months (historical range)
        - 💰 **Budget Overruns:** Unexpected cost increases lead to project losses
        - 🤝 **Negotiation Disadvantage:** Contractors lack data-driven forecasting tools
        - ⏰ **Timing Risk:** Poor timing in material procurement affects margins
        """)
    
    
    with col2:
        st.info("""
        Example Based on This Model:
        
        Project: 10 million dollar construction    
        Steel costs: 15% = 1.5 million dollars
        
        12-month forecast range:
        • Pessimistic: -13% (LSTM model)
        • Optimistic: +14% (Prophet model)
        
        Steel budget range: 1.3M to 1.7M
        Price exposure: +/- 200K dollars
        """)
    
    st.markdown("---")
    
    # Solution
    st.subheader("💡 Our Solution: AI-Powered Price Forecasting")
    
    st.markdown("""
    This project leverages **machine learning and statistical modeling** to forecast steel prices up to 3 years ahead, 
    enabling construction professionals to make data-driven decisions.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background-color: rgba(255,99,71,0.1); border-radius: 8px;'>
            <h3 style='color: #ff6347;'>📊 9 Models</h3>
            <p>Baseline to Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background-color: rgba(30,144,255,0.1); border-radius: 8px;'>
            <h3 style='color: #1e90ff;'>🎯 Scenario Analysis</h3>
            <p>Risk-Based Planning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background-color: rgba(50,205,50,0.1); border-radius: 8px;'>
            <h3 style='color: #32cd32;'>📈 43 Years Data</h3>
            <p>1982-2025 History</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background-color: rgba(255,165,0,0.1); border-radius: 8px;'>
            <h3 style='color: #ffa500;'>🧠 LSTM + ARIMA</h3>
            <p>Deep Learning + Statistical</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technical Approach
    st.subheader("🔬 Technical Approach")
    
    tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Models", "💼 Application"])
    
    with tab1:
        st.markdown("""
        **Data Source:** U.S. Bureau of Labor Statistics (via FRED)
        - **Series:** WPU101704 - Producer Price Index for Hot Rolled Steel
        - **Period:** June 1982 - September 2025 (520 monthly observations)
        - **Frequency:** Monthly
        - **Base Index:** June 1982 = 100
        
        **Data Quality:**
        - ✅ Official government statistics
        - ✅ No missing values
        - ✅ Consistent measurement methodology
        - ✅ Real-time market indicator
        """)
        
        # Show sample data
        st.markdown("**Sample Data Preview:**")
        st.dataframe(df.tail(10)[['observation_date', 'WPU101704']], use_container_width=True)
    
    with tab2:
        st.markdown("""
        **Model Portfolio (9 Models Evaluated):**
        
        | Category | Models | Best RMSE |
        |----------|--------|-----------|
        | **Deep Learning** | LSTM | **6.19** ⭐ |
        | **Statistical** | ARIMA, SES, Holt's, Holt-Winters | 12.49 |
        | **ML-Based** | Prophet | 36.60 |
        | **Baseline** | Naive, Seasonal Naive, MA-12 | 8.93 |
        
        **Recommended Models:**
        - **Production:** ARIMA (realistic trends, any horizon)
        - **High Accuracy:** LSTM (best test performance, ≤12 months)
        - **Scenario Planning:** Multi-model ensemble (ARIMA + Naive + Prophet)
        """)
    
    with tab3:
        st.markdown("""
        **Real-World Use Cases:**
        
        **1. Budget Planning** 📊
        - Use forecasts to set material budgets for upcoming projects
        - Apply pessimistic scenarios for contingency planning
        
        **2. Procurement Strategy** 🛒
        - Identify optimal timing for bulk steel purchases
        - Lock in prices when forecasts predict increases
        
        **3. Bid Preparation** 💰
        - Include data-driven contingencies in project bids
        - Adjust pricing based on 6-12 month forecasts
        
        **4. Supplier Negotiation** 🤝
        - Support negotiation with objective price forecasts
        - Show multiple scenarios to suppliers for better terms
        
        **5. Risk Management** ⚠️
        - Quantify price uncertainty (±X% range)
        - Recommend contingency % based on volatility
        """)
    
    st.markdown("---")
    
    # Business Impact
    st.subheader("📈 Business Impact")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **Potential Benefits for Contractors:**
        
        ✅ **Better Planning**  
        Make informed decisions based on data, not guesswork
        
        ✅ **Risk Mitigation**  
        Quantify price uncertainty to set appropriate contingencies
        
        ✅ **Competitive Advantage**  
        Data-backed bids vs. competitors using intuition
        
        ✅ **Time Savings**  
        Instant scenario analysis vs. manual spreadsheet work
        """)
        
    with col2:
        st.info("""
        **Project Statistics:**
        
        📊 **520 months** of historical data  
        (June 1982 - September 2025)
        
        🧠 **9 algorithms** trained & evaluated  
        (From Baseline to Deep Learning)
        
        🎯 **1.73% MAPE** Test Accuracy  
        (Mean Absolute Percentage Error)
        
        📈 **1-36 months** Forecast Horizons  
        (Short-term to Long-term planning)
        
        ⚡ **<1 second** Loading Time  
        (GPU-accelerated pre-computation)
        """)
    
    st.markdown("---")
    
    # How to Use
    st.subheader("🚀 How to Use This Dashboard")
    
    st.markdown("""
    **Step-by-Step Guide:**
    
    1. **📈 Overview** - Understand historical price trends and statistics
    2. **🔍 Model Comparison** - See which models perform best on test data
    3. **🔮 Forecast** - Generate price predictions:
       - **Scenario Analysis** (Recommended): Get pessimistic, expected, and optimistic forecasts
       - **Single Model**: Choose specific model for detailed forecast
    4. **📚 Documentation** - Learn about models and methodology
    
    **💡 Pro Tip for Contractors:**
    Start with **Scenario Analysis** for any upcoming project. Use the pessimistic forecast 
    for budget planning, expected for baseline, and optimistic to understand upside potential.
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: rgba(100,100,100,0.1); border-radius: 10px;'>
        <h4>🎓 RuangGuru AI Engineering Bootcamp - Final Project</h4>
        <p><strong>Time Series Forecasting for Construction Material Price Prediction</strong></p>
        <p>Developed by Felix Kho | GitHub: <a href='https://github.com/felixkhoiscoding'>felixkhoiscoding</a></p>
        <p>Data Source: <a href='https://fred.stlouisfed.org/series/WPU101704'>FRED - WPU101704</a> | Last Updated: September 2025</p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 1: Overview
elif page == "📈 Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", len(df))
    with col2:
        st.metric("Date Range", f"{df['observation_date'].min().year}-{df['observation_date'].max().year}")
    with col3:
        st.metric("Current Value", f"{df['WPU101704'].iloc[-1]:.2f}")
    with col4:
        st.metric("43-Year Change", f"+{((df['WPU101704'].iloc[-1]/df['WPU101704'].iloc[0]-1)*100):.1f}%")
    
    # Time series plot
    st.subheader("Historical Price Index")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['observation_date'],
        y=df['WPU101704'],
        mode='lines',
        name='PPI',
        line=dict(color='steelblue', width=2)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Value (Jun 1982 = 100)",
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Key Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Descriptive Statistics:**")
        stats_df = df['WPU101704'].describe().to_frame('Value')
        st.dataframe(stats_df, use_container_width=True)
    with col2:
        st.write("**Recent Performance (Last 12 Months):**")
        recent = df.tail(12)
        st.metric("Mean", f"{recent['WPU101704'].mean():.2f}")
        st.metric("Volatility (Std)", f"{recent['WPU101704'].std():.2f}")
        st.metric("Min-Max Range", f"{recent['WPU101704'].min():.2f} - {recent['WPU101704'].max():.2f}")

# PAGE 2: Model Comparison
elif page == "🔍 Model Comparison":
    st.header("Model Performance Comparison")
    
    # Results from comprehensive testing - ALL 9 MODELS
    results_data = {
        'Model': ['Naive', 'LSTM', 'ARIMA', 'SES', 'MA-12', 'Seasonal Naive', 
                  'Prophet', 'Holt-Winters', "Holt's"],
        'RMSE': [8.93, 6.19, 12.49, 17.61, 20.16, 23.44, 36.60, 40.71, 44.62],
        'MAPE': [2.89, 1.73, 4.75, 5.98, 7.07, 7.84, 13.80, 12.59, 13.51],
        'Type': ['Baseline', 'Advanced', 'Advanced', 'Baseline', 'Baseline', 
                 'Baseline', 'Advanced', 'Baseline', 'Baseline']
    }
    results_df = pd.DataFrame(results_data).sort_values('RMSE')
    
    st.subheader("📊 Test Set Performance (Oct 2024 - Sep 2025)")
    
    # Metrics table
    st.dataframe(
        results_df.style.background_gradient(subset=['RMSE', 'MAPE'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Best models highlight
    st.success(f"🏆 **Best Test Performance:** {results_df.iloc[0]['Model']} (RMSE: {results_df.iloc[0]['RMSE']:.2f})")
    st.info("""
    **Model Recommendations:**
    - **For Production Forecasting:** Use **ARIMA** - provides realistic trends, works for any horizon
    - **For Test Accuracy (≤12 months):** LSTM has best performance but limited to short horizons
    - **For Simplicity:** Naive provides baseline but shows flat forecasts (not user-friendly)
    """)
    
    # Visualization
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(results_df, x='Model', y='RMSE', color='Type',
                    title='RMSE Comparison (Lower is Better)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(results_df, x='Model', y='MAPE', color='Type',
                    title='MAPE Comparison (Lower is Better)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric Explanations
    st.subheader("📖 Understanding the Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **RMSE - Root Mean Squared Error**
        
        Measures prediction error in the same units as your data (index points). Penalizes large errors more heavily.
        
        - **Formula:** Square root of average squared errors
        - **What it means:** If RMSE = 8.93, typical prediction errors are around 9 points
        - **Lower is better:** RMSE of 6 beats RMSE of 12
        - **Sensitive to outliers:** One big mistake hurts RMSE a lot
        
        **Example:** Errors of [5, 5, 5, 5] give RMSE=5, but [0, 0, 0, 10] gives RMSE=5 too (different patterns, same RMSE).
        """)
    
    with col2:
        st.markdown("""
        **MAPE - Mean Absolute Percentage Error**
        
        Measures prediction error as a percentage of actual values.
        
        - **Formula:** Average of |error/actual| × 100%
        - **What it means:** If MAPE = 2.89%, predictions are typically off by ~3%
        - **Lower is better:** MAPE of 2% beats MAPE of 5%
        - **Easy to interpret:** 1.73% = very accurate, 15% = poor
        
        **Example:** Actual=270, Predicted=280 → Error=10 → MAPE = |10/270| × 100% = 3.7%
        """)
    
    st.info("💡 **For Construction Contractors:** MAPE is easier to understand (% error), while RMSE shows error magnitude in actual price points. Both measure accuracy - lower values mean better forecasts.")


# PAGE 3: Forecast
elif page == "🔮 Forecast":
    st.header("Future Price Forecast")
    
    # Add view mode selection
    view_mode = st.radio(
        "Select view:",
        ["📊 Scenario Analysis (Recommended for Risk Planning)", "🎯 Single Model Forecast"],
        help="Scenario Analysis shows multiple forecasts for risk assessment"
    )
    
    # USER INPUT: Forecast horizon
    forecast_months = st.slider(
        "Select forecast horizon (months):",
        min_value=1,
        max_value=36,
        value=FORECAST_HORIZON,
        help="Number of months to forecast ahead"
    )
    
    if view_mode == "📊 Scenario Analysis (Recommended for Risk Planning)":
        # SCENARIO ANALYSIS MODE
        st.info("🏗️ **For Construction Contractors:** This view shows multiple price scenarios to support budget planning, risk assessment, and supplier negotiation.")
        
        if st.button("Generate Scenario Analysis", type="primary"):
            # Add disclaimer about pre-computed forecasts  
            st.info("📊 **Note:** Forecasts are pre-computed using GPU-accelerated training for instant loading. Based on data through September 2025.")
            
            with st.spinner("Loading scenarios..."):
                
                # Create future dates
                last_date = df['observation_date'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )
                
                # Load pre-computed scenarios
                scenarios = {}
                fitted_all = {}
                
                # 1. Pessimistic (ARIMA)
                arima_forecast, arima_fitted = load_precomputed_forecast('arima', forecast_months)
                if arima_forecast is not None:
                    scenarios['Pessimistic (ARIMA)'] = arima_forecast
                    fitted_all['ARIMA'] = arima_fitted
                else:
                    # Fallback to real-time
                    try:
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        model = SARIMAX(df['WPU101704'], order=(1,1,1))
                        fitted = model.fit(disp=False)
                        scenarios['Pessimistic (ARIMA)'] = fitted.forecast(steps=forecast_months).values
                        fitted_all['ARIMA'] = fitted.fittedvalues
                    except:
                        scenarios['Pessimistic (ARIMA)'] = None
                
                # 2. Expected (Naive)
                naive_forecast, naive_fitted = load_precomputed_forecast('naive', forecast_months)
                if naive_forecast is not None:
                    scenarios['Expected (Naive)'] = naive_forecast
                    fitted_all['Naive'] = naive_fitted
                else:
                    # Fallback
                    last_value = df['WPU101704'].iloc[-1]
                    scenarios['Expected (Naive)'] = np.full(forecast_months, last_value)
                    fitted_all['Naive'] = df['WPU101704'].shift(1)
                
                # 3. Optimistic (Prophet)
                prophet_forecast, prophet_fitted = load_precomputed_forecast('prophet', forecast_months)
                if prophet_forecast is not None:
                    scenarios['Optimistic (Prophet)'] = prophet_forecast
                    fitted_all['Prophet'] = prophet_fitted
                else:
                    # Fallback
                    try:
                        from prophet import Prophet
                        prophet_data = pd.DataFrame({
                            'ds': df['observation_date'],
                            'y': df['WPU101704']
                        })
                        model = Prophet(yearly_seasonality=True)
                        model.fit(prophet_data)
                        future = pd.DataFrame({'ds': future_dates})
                        forecast = model.predict(future)
                        scenarios['Optimistic (Prophet)'] = forecast['yhat'].values
                        fitted_all['Prophet'] = model.predict(prophet_data)['yhat'].values
                    except:
                        scenarios['Optimistic (Prophet)'] = None
                
                # 4. LSTM (only for <=12 months) - Load from pre-computed
                if forecast_months <= 12:
                    lstm_forecast, lstm_fitted = load_precomputed_forecast('lstm', forecast_months)
                    if lstm_forecast is not None:
                        scenarios['Best Performance (LSTM)'] = lstm_forecast
                        fitted_all['LSTM'] = lstm_fitted
                    # No fallback for LSTM - too slow
                
                # Current price
                current_price = df['WPU101704'].iloc[-1]
                
                # Scenario Summary Table
                st.subheader("📋 Price Scenario Summary")
                
                summary_data = []
                for name, values in scenarios.items():
                    if values is not None:
                        end_price = values[-1]
                        change_pct = ((end_price - current_price) / current_price) * 100
                        summary_data.append({
                            'Scenario': name.split(' (')[0],
                            'Model': name.split('(')[1].rstrip(')'),
                            'Current Price': f"${current_price:.2f}",
                            f'{forecast_months}-Month Price': f"${end_price:.2f}",
                            'Change': f"{change_pct:+.1f}%"
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Risk Metrics
                st.subheader("⚠️ Risk Assessment")
                
                col1, col2, col3 = st.columns(3)
                
                valid_scenarios = [v for v in scenarios.values() if v is not None]
                if len(valid_scenarios) >= 2:
                    all_end_prices = [v[-1] for v in valid_scenarios]
                    price_range = max(all_end_prices) - min(all_end_prices)
                    range_pct = (price_range / current_price) * 100
                    
                    with col1:
                        st.metric(
                            "Price Range",
                            f"${min(all_end_prices):.2f} - ${max(all_end_prices):.2f}",
                            f"±{range_pct/2:.1f}%"
                        )
                    
                    with col2:
                        # Better contingency calculation
                        if range_pct < 5:
                            contingency = 5  # Low risk - small buffer
                        elif range_pct < 10:
                            contingency = 10  # Medium-low risk
                        elif range_pct < 20:
                            contingency = range_pct  # Medium risk - match range
                        else:
                            contingency = min(range_pct, 25)  # High risk - cap at 25%
                        
                        st.metric(
                            "Recommended Contingency",
                            f"{contingency:.0f}%",
                            "Budget buffer"
                        )
                    
                    with col3:
                        if range_pct < 5:
                            risk_level = "LOW"
                            risk_color = "🟢"
                        elif range_pct < 15:
                            risk_level = "MEDIUM"
                            risk_color = "🟡"
                        else:
                            risk_level = "HIGH"
                            risk_color = "🔴"
                        
                        st.metric(
                            "Price Uncertainty",
                            f"{risk_color} {risk_level}",
                            f"{range_pct:.1f}% spread"
                        )
                
                # Contractor Recommendations
                st.subheader("💼 Contractor Action Items")
                
                recommendations = []
                if range_pct < 5:
                    recommendations.append("✅ **Low Risk:** Prices expected to remain stable. Standard contingency (5-10%) sufficient.")
                elif range_pct < 15:
                    recommendations.append(f"⚠️ **Medium Risk:** Price volatility expected. Include {contingency:.0f}% contingency in bids.")
                else:
                    recommendations.append(f"🔴 **High Risk:** Significant price uncertainty. Strongly recommend {contingency:.0f}% contingency and fixed-price contracts with suppliers.")
                
                recommendations.append(f"📊 **Negotiation Basis:** Use pessimistic forecast (${all_end_prices[0] if scenarios['Pessimistic (ARIMA)'] is not None else 'N/A':.2f}) as negotiation baseline with suppliers.")
                recommendations.append(f"💰 **Budget Planning:** Budget for optimistic scenario (${max(all_end_prices):.2f}) to avoid cost overruns.")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Visualization - All scenarios with FITTED VALUES
                st.subheader("📈 Scenario Comparison Chart")
                st.info("💡 **Chart shows:** Thin dashed lines = how each model fits historical data | Thick dashed lines = future forecasts")
                
                fig = go.Figure()
                
                # Historical ACTUAL data (bold gray line)
                fig.add_trace(go.Scatter(
                    x=df['observation_date'],
                    y=df['WPU101704'],
                    mode='lines',
                    name='Actual (Historical)',
                    line=dict(color='gray', width=3),
                    legendgroup='actual'
                ))
                
                # Generate FITTED VALUES for each model (historical period)
                # Use pre-loaded fitted values from fitted_all dictionary
                fitted_values = fitted_all.copy()
                
                # Plot FITTED values (thin dashed lines)
                fitted_colors = {'ARIMA': 'red', 'Naive': 'blue', 'Prophet': 'green', 'LSTM': 'orange'}
                for model_name, values in fitted_values.items():
                    if values is not None:
                        fig.add_trace(go.Scatter(
                            x=df['observation_date'],
                            y=values,
                            mode='lines',
                            name=f'{model_name} Fit (Historical)',
                            line=dict(color=fitted_colors[model_name], width=1, dash='dot'),
                            opacity=0.7,
                            legendgroup=model_name.lower()
                        ))
                
                # Plot FORECASTS (thick dashed lines)
                forecast_colors = {
                    'Pessimistic (ARIMA)': 'red', 
                    'Expected (Naive)': 'blue', 
                    'Optimistic (Prophet)': 'green',
                    'Best Performance (LSTM)': 'orange'
                }
                for name, values in scenarios.items():
                    if values is not None:
                        model_short = name.split(' (')[1].rstrip(')')
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=values,
                            mode='lines+markers',
                            name=f'{model_short} Forecast',
                            line=dict(color=forecast_colors.get(name, 'purple'), width=3, dash='dash'),
                            marker=dict(size=4),
                            legendgroup=model_short.lower()
                        ))
                
                # Add vertical line at forecast start (VISIBLE ON BOTH THEMES)
                fig.add_vline(
                    x=pd.to_datetime(df['observation_date'].max()).timestamp() * 1000,
                    line_dash="dash",
                    line_color="cyan",  # Bright cyan - visible on both dark and light themes
                    line_width=3,
                    opacity=1  # Full opacity for maximum visibility
                )
                
                # Add annotation manually
                fig.add_annotation(
                    x=df['observation_date'].max(),
                    y=1.02,
                    yref="paper",
                    text="◄ Historical Data | Forecast ►",
                    showarrow=False,
                    font=dict(size=12, color="cyan", family="Arial Black"),
                    bgcolor="rgba(0,0,0,0.7)",  # Dark semi-transparent background
                    bordercolor="cyan",
                    borderwidth=2,
                    xanchor="center"
                )
                
                fig.update_layout(
                    title=f"Historical Fit + {forecast_months}-Month Forecast Comparison",
                    xaxis_title="Date",
                    yaxis_title="Price Index",
                    hovermode='x unified',
                    height=700,
                    # Removed template="plotly_dark" to allow theme switching
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(50,50,50,0.9)",
                        bordercolor="white",
                        borderwidth=1,
                        font=dict(color="white")
                    ),
                    hoverlabel=dict(
                        bgcolor="rgba(50,50,50,0.9)",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation guide
                with st.expander("📖 How to Read This Chart"):
                    st.markdown("""
                    **Understanding the Visualization:**
                    
                    - **Gray thick line:** Actual historical prices (what really happened)
                    - **Thin dotted lines:** How each model "fits" or explains historical data
                        - Closer to gray = better historical fit
                        - Shows what patterns each model learned
                    - **Thick dashed lines:** Future forecasts from each model
                    - **Black vertical line:** Divides past from forecast
                    
                    **What to Look For:**
                    1. **Historical Fit Quality:** Does the model's fitted line follow actual prices well?
                    2. **Transition:** How does the model move from fit to forecast?
                    3. **Forecast Divergence:** How much do future predictions differ?
                    
                    **Model Behavior:**
                    - **ARIMA (Red):** Captures trends and slowly reverts to mean
                    - **Naive (Blue):** Simple persistence - follows with 1-month lag
                    - **Prophet (Green):** Captures seasonality and growth trends
                    - **LSTM (Orange):** Deep learning patterns (only shown for ≤12 month forecasts)
                    """)
                
                # Download all scenarios
                scenario_df = pd.DataFrame({
                    'Date': future_dates.strftime('%Y-%m')
                })
                for name, values in scenarios.items():
                    if values is not None:
                        scenario_df[name] = values
                
                csv = scenario_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Scenarios (CSV)",
                    data=csv,
                    file_name=f"price_scenarios_{forecast_months}months.csv",
                    mime="text/csv"
                )
    
    else:
        # SINGLE MODEL MODE (original)
        
        model_options = ["ARIMA (Recommended)", "Naive (Simple)", "Prophet"]
        
        # Only show LSTM for <=12 months
        if forecast_months <= 12:
            model_options.insert(1, "LSTM (Deep Learning)")
        
        model_choice = st.selectbox(
            "Select forecasting model:",
            model_options
        )
        
        # Model explanations
        if "ARIMA" in model_choice:
            st.info("📊 **ARIMA** provides statistically sound forecasts with trend analysis. Best for most use cases.")
        elif "LSTM" in model_choice:
            st.info("🧠 **LSTM** uses deep learning for pattern recognition. Best performance on test data but limited to short horizons.")
        elif "Naive" in model_choice:
            st.info("📌 **Naive** assumes future = current value. Simple baseline, best RMSE on test set, but may appear too simplistic.")
        else:
            st.info("📈 **Prophet** handles seasonality and holidays well. Developed by Meta.")
        
        # Warning for LSTM limitation
        if forecast_months > 12:
            st.warning("""⚠️ **LSTM not available for horizons > 12 months** due to error accumulation in iterative forecasting. 
            Use **ARIMA** for longer forecasts.""")
        
        if st.button("Generate Forecast", type="primary"):
            # Add disclaimer
            st.info("📊 **Note:** Forecasts are pre-computed using GPU-accelerated training for instant loading. Based on data through September 2025.")
            
            with st.spinner("Loading forecast..."):
                
                # Create future dates
                last_date = df['observation_date'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )
                
                # Map model choice to file name
                model_map = {
                    "ARIMA (Recommended)": "arima",
                    "Naive (Simple)": "naive",
                    "Prophet": "prophet",
                    "LSTM (Deep Learning)": "lstm"
                }
                
                # Get model name for CSV loading
                model_file_name = None
                for key, value in model_map.items():
                    if key in model_choice:
                        model_file_name = value
                        break
                
                # Load pre-computed forecast
                forecast_values, fitted_values_arr = load_precomputed_forecast(model_file_name, forecast_months)
                
                # Set model color
                color_map = {"arima": "red", "naive": "blue", "prophet": "green", "lstm": "orange"}
                model_color = color_map.get(model_file_name, "red")
                
                # Create forecast dataframe
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': forecast_values
                })
                
                # Visualization with FITTED VALUES
                st.info("💡 **Chart shows:** Thin dotted line = model's fit to historical data | Thick dashed line = future forecast")
                fig = go.Figure()
                
                # Historical ACTUAL data (gray thick line)
                fig.add_trace(go.Scatter(
                    x=df['observation_date'],
                    y=df['WPU101704'],
                    mode='lines',
                    name='Actual (Historical)',
                    line=dict(color='gray', width=3)
                ))
                
                # FITTED values (thin dotted line)
                if fitted_values_arr is not None:
                    fig.add_trace(go.Scatter(
                        x=df['observation_date'],
                        y=fitted_values_arr,
                        mode='lines',
                        name=f'{model_choice.split(" (")[0]} Fit',
                        line=dict(color=model_color, width=1, dash='dot'),
                        opacity=0.7
                    ))
                
                # FORECAST (thick dashed line)
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Forecast'],
                    mode='lines+markers',
                    name=f'{model_choice.split(" (")[0]} Forecast',
                    line=dict(color=model_color, width=3, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Add cyan divider line
                fig.add_vline(
                    x=pd.to_datetime(df['observation_date'].max()).timestamp() * 1000,
                    line_dash="dash",
                    line_color="cyan",
                    line_width=3,
                    opacity=1
                )
                
                fig.add_annotation(
                    x=df['observation_date'].max(),
                    y=1.02,
                    yref="paper",
                    text="◄ Historical Data | Forecast ►",
                    showarrow=False,
                    font=dict(size=12, color="cyan", family="Arial Black"),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="cyan",
                    borderwidth=2,
                    xanchor="center"
                )
                
                fig.update_layout(
                    title=f"Historical Fit + {forecast_months}-Month Forecast using {model_choice}",
                    xaxis_title="Date",
                    yaxis_title="Index Value",
                    hovermode='x unified',
                    height=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("📋 Forecast Values")
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m')
                st.dataframe(forecast_df, use_container_width=True)
                
                # Download
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"ppi_forecast_{forecast_months}months.csv",
                    mime="text/csv"
                )

# PAGE 4: Documentation
else:
    st.header("📚 Project Documentation")
    
    st.markdown("""
    ## Time-Series Forecasting Project
    
    ### Dataset
    - **Source:** FRED (Federal Reserve Economic Data)
    - **Series:** WPU101704 - Producer Price Index for Hot Rolled Steel
    - **Period:** June 1982 - September 2025 (520 months)
    - **Frequency:** Monthly
    - **Base Index:** June 1982 = 100
    
    ### Methodology
    
    **1. Data Analysis**
    - Exploratory Data Analysis (EDA)
    - Stationarity testing (ADF, KPSS)
    - Seasonality decomposition
    
    **2. Models Tested (9 Total)**
    - **Baseline Models:** Naive, Seasonal Naive, MA-12, SES, Holt's, Holt-Winters
    - **Statistical Models:** ARIMA
    - **ML-Based Models:** Prophet (Meta)
    - **Deep Learning Models:** LSTM (Neural Network)
    
    **3. Evaluation Metrics**
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    
    ### Key Findings
    
    - 🥇 **Best Test Performance:** LSTM (RMSE: 6.19, MAPE: 1.73%)
    - 🥈 **Runner-up:** Naive (RMSE: 8.93) - but too simplistic for production
    - 📊 **Test Period:** October 2024 - September 2025 (12 months)
    - ⚠️ **LSTM Limitation:** Reliable only for ≤12 month forecasts
    - ✅ **Production Recommendation:** **ARIMA** (realistic trends + any horizon)
    
    ### Model Selection Guide
    
    | Horizon | Recommended | Alternative | Avoid |
    |---------|------------|-------------|-------|
    | 1-12 months | ARIMA or LSTM | Prophet | - |
    | 13-24 months | ARIMA | Prophet | LSTM (error accumulation) |
    | 25+ months | ARIMA | Prophet | LSTM, Naive (flat) |
    
    **Why ARIMA for Production:**
    - Shows realistic upward/downward trends
    - Statistically validated methodology  
    - Works reliably for any time horizon
    - Users take it seriously (not "too naive")
    
    ### Pre-Computed Forecasts
    
    **For optimal user experience and cloud deployment:**
    - All forecasts pre-generated locally using GPU (LSTM training)
    - Saved as CSV files for instant loading (<1 second)
    - Enables deployment on CPU-only Streamlit Cloud
    - Reproducible results using random seed (seed=42)
    - Update monthly when new data becomes available
    
    ### Technical Stack
    - **Language:** Python 3.13
    - **Libraries:** pandas, numpy, statsmodels, prophet, tensorflow
    - **Visualization:** plotly, matplotlib, seaborn
    - **Deployment:** Streamlit Cloud
    
    ### Usage
    
    ```python
    # Run locally
    streamlit run streamlit_app.py
    
    # Regenerate forecasts (optional, requires GPU for LSTM)
    python generate_forecasts.py
    ```
    
    ### Project Structure
    ```
    Final Project/
    ├── streamlit_app.py      # Main dashboard
    ├── generate_forecasts.py # Forecast generation
    ├── src/                  # Source modules
    ├── data/                 # Raw, processed, and forecasts
    ├── notebooks/            # Jupyter analysis
    ├── config/               # Configuration
    └── requirements.txt      # Dependencies
    ```
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**PPI Forecasting Dashboard**  
Version 2.0  
Built with Streamlit

**Developer:**  
Felix Kho  
[GitHub](https://github.com/felixkhoiscoding) | [LinkedIn](https://www.linkedin.com/in/felixkho/)

**Model Recommendations:**
- **Production:** ARIMA  
- **Best Test:** LSTM (≤12mo)
- **Simplest:** Naive
""")


