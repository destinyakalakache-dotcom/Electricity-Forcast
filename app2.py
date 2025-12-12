import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.express as px

# --- 1. CONFIGURATION AND INITIAL LOADING ---
# Define file paths (Adjust these paths as necessary)
UK_HOLIDAYS_PATH = 'uk_holidays.csv'
SCALER_PATH = 'saved_models/scaler.joblib' 
MODEL_PATH = 'saved_models/xgboost_model.joblib' 

# Features the model was trained on (must match exactly)
FEATURES_TO_KEEP = [
    'settlement_period', 
    'embedded_wind_generation', 
    'embedded_solar_generation', 
    'embedded_wind_capacity',
    'embedded_solar_capacity', 
    'non_bm_stor', 
    'pump_storage_pumping', 
    'ifa_flow', 
    'ifa2_flow', 
    'britned_flow', 
    'moyle_flow', 
    'east_west_flow', 
    'nemo_flow', 
    'england_wales_demand_lag_1', 
    'england_wales_demand_lag_2', 
    'england_wales_demand_lag_24', 
    'england_wales_demand_lag_48', 
    'england_wales_demand_lag_336', 
    'england_wales_demand_rolling_mean_24',
    'england_wales_demand_rolling_std_24', 
    'england_wales_demand_rolling_mean_48', 
    'england_wales_demand_rolling_std_48', 
    'england_wales_demand_rolling_mean_336', 
    'england_wales_demand_rolling_std_336', 
    'is_holiday', 
    'hour_sin', 
    'hour_cos', 
    'day_of_week_sin', 
    'day_of_week_cos', 
    'month_sin', 
    'month_cos'
]

# Use Streamlit's caching for efficient data and model loading
@st.cache_resource
def load_resources():
    try:
        df_holidays = pd.read_csv(UK_HOLIDAYS_PATH)
        uk_holidays = set(pd.to_datetime(df_holidays['date']).dt.date)
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        return uk_holidays, scaler, model
    except FileNotFoundError as e:
        st.error(f"Error loading resources. Please ensure all files are correctly placed. Missing: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        st.stop()

# Load all resources once
uk_holidays, scaler, model = load_resources()

# --- 2. PREPROCESSING FUNCTIONS (Kept as before) ---
def generate_time_features(dt_index, uk_holidays):
    """Generates cyclical and holiday features for a given datetime index."""
    hour = dt_index.hour + dt_index.minute / 60.0 
    day_of_week = dt_index.dayofweek
    month = dt_index.month
    
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    is_holiday = 1 if dt_index.date() in uk_holidays else 0
    return {
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_of_week_sin': day_of_week_sin,
        'day_of_week_cos': day_of_week_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_holiday': is_holiday
    }

def generate_dummy_lagged_data(start_dt, max_lag=336):
    """Generates a dummy historical demand series for initial lagged and rolling features."""
    np.random.seed(42)
    periods = max_lag + 1
    dates = pd.date_range(end=start_dt - timedelta(minutes=30), periods=periods, freq='30T')
    dummy_demands = 20000 + 5000 * np.sin(np.linspace(0, 10, periods)) + np.random.normal(0, 1000, periods)
    df_dummy = pd.DataFrame({'datetime': dates, 'england_wales_demand': dummy_demands}).set_index('datetime')
    return df_dummy

def compute_lagged_and_rolling_features(df_history):
    """Computes lagged and rolling features from the provided history DataFrame."""
    lag_periods = [1, 2, 24, 48, 336]
    rolling_windows = [24, 48, 336]
    demand_lags = {}
    demand_rolling = {}
    
    for lag in lag_periods:
        demand_lags[f'england_wales_demand_lag_{lag}'] = df_history['england_wales_demand'].iloc[-lag] if len(df_history) >= lag else 0.0

    for window in rolling_windows:
        if len(df_history) >= window:
            rolling_slice = df_history['england_wales_demand'].tail(window)
            demand_rolling[f'england_wales_demand_rolling_mean_{window}'] = rolling_slice.mean()
            demand_rolling[f'england_wales_demand_rolling_std_{window}'] = rolling_slice.std()
        else:
            demand_rolling[f'england_wales_demand_rolling_mean_{window}'] = 0.0
            demand_rolling[f'england_wales_demand_rolling_std_{window}'] = 0.0

    return dict(demand_lags, **demand_rolling)


# --- 3. STREAMLIT APP LAYOUT & SIDEBAR NAVIGATION ---
st.set_page_config(page_title="UK Electricity Demand Predictor", layout="wide")

# =========================================================================
# === SIDEBAR: NAVIGATION SELECTOR ===
# =========================================================================
with st.sidebar:
    st.title("UK Demand Forecast App")
    
    # Navigation Selector
    page = st.radio(
        "Navigate Application",
        ["⚡ Prediction Tool", "🎓 Project Details"],
        captions=["Generate the 24-hour forecast.", "View student info and project description."],
        index=0 # Default to Prediction Tool
    )
    st.markdown("---")

# =========================================================================
# === MAIN PANE: CONDITIONAL RENDERING ===
# =========================================================================

if page == "🎓 Project Details":
    
    # ---------------------------------------------
    # PROJECT DETAILS PAGE
    # ---------------------------------------------
    st.header("🎓 Project Details: UK Electricity Demand Forecasting")
    st.markdown("This section provides the context, structure, and academic information related to the application.")
    
    st.markdown("---")
    
    st.subheader("Student Information")
    
    # 
    st.image("daniel.jpg", caption="Student Name", width=200) 
    
    st.markdown("""
        *   **Name:** Daniel Chetachukwu Eneh
        *   **Student ID:** 2329276
        *   **Department:** MSc Data Science
        *   **Module:** CMM500
    """)

    st.subheader("Project Description 💡")
    st.markdown("""
        This application demonstrates a **24-hour ahead forecasting model** for **England & Wales Electricity Demand (MW)**. 
        
        The model uses a **XGBoost** algorithm trained on historical time series data. The forecasting task is challenging as it requires capturing complex non-linear relationships and temporal dependencies.

        ### Model Inputs
        The model integrates three main categories of features:
        1.  **Exogenous Factors:** Generation forecasts (Wind, Solar), Interconnector flows (IFA, BritNed, etc.), and Storage levels.
        2.  **Temporal Features:** Cyclical components derived from the date/time (hour, day of week, month) and public holidays, encoded using **sin/cos transformations** to preserve continuity.
        3.  **Endogenous (Lagged) Features:** Past observed demand values (e.g., lag 1, lag 48, lag 336) and rolling statistics (mean, standard deviation) of demand to capture recent trends and volatility.

        ### Forecasting Methodology
        The forecast is generated **recursively** over a 48-period horizon (24 hours). This means the prediction for period $t$ is fed back as the required lagged feature for predicting period $t+1$, mimicking the real-time operational use of the model. Initial lags are derived from simulated data for demonstration purposes.
    """)
    
elif page == "⚡ Prediction Tool":
    
    # ---------------------------------------------
    # PREDICTION TOOL PAGE 
    # ---------------------------------------------
    st.title("💡 Prediction Tool: UK Electricity Demand Forecast")
    st.markdown("Use the inputs below to generate a 24-hour forecast for the **England & Wales Demand (MW)**.")

    st.subheader("--- ⚙️ Input Parameters ---")

    # --- Time Inputs ---
    st.header("🕰️ Time Inputs")
    col1, col2 = st.columns(2)
    with col1:
        settlement_date = st.date_input("Starting Settlement Date", datetime.now().date())
    with col2:
       
        settlement_period = 1
        
    start_dt = pd.to_datetime(settlement_date) + (settlement_period - 1) * timedelta(minutes=30)
    st.info(f"The 24-hour forecast will begin at: **{start_dt.strftime('%Y-%m-%d %H:%M:%S')}**")

    # --- Generation/Storage Inputs ---
    st.header("🌬️ Generation & Capacity (MW)")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        embedded_wind_generation = st.number_input("Embedded Wind Gen", value=1500.0, min_value=0.0)
    with col4:
        embedded_solar_generation = st.number_input("Embedded Solar Gen", value=500.0, min_value=0.0)
    with col5:
        embedded_wind_capacity = st.number_input("Embedded Wind Capacity", value=15000.0, min_value=0.0)
    with col6:
        embedded_solar_capacity = st.number_input("Embedded Solar Capacity", value=12000.0, min_value=0.0)

    st.header("🔋 Storage & Pumping (MW)")
    col7, col8 = st.columns(2)
    with col7:
        non_bm_stor = st.number_input("Non-BM Storage", value=100.0)
    with col8:
        pump_storage_pumping = st.number_input("Pump Storage Pumping", value=-50.0)

    # --- Interconnector Inputs ---
    st.header("🌐 Interconnector Flows (MW)")
    col9, col10, col11 = st.columns(3)
    with col9:
        ifa_flow = st.number_input("IFA Flow (France)", value=500.0)
        ifa2_flow = st.number_input("IFA2 Flow (France)", value=100.0)
    with col10:
        britned_flow = st.number_input("BritNed Flow (NL)", value=500.0)
        moyle_flow = st.number_input("Moyle Flow (NI)", value=-100.0)
    with col11:
        east_west_flow = st.number_input("East-West Flow (IRL)", value=50.0)
        nemo_flow = st.number_input("NEMO Flow (Belgium)", value=250.0)

    st.markdown("---")

    # --- 4. PREDICTION LOGIC ---
    if st.button("🚀 Generate 24-Hour Forecast"):
        st.subheader("Processing Inputs and Generating Forecast...")

        # 1. Collect Exogenous Features (User Provided)
        exogenous_dict = {
            'embedded_wind_generation': embedded_wind_generation,
            'embedded_solar_generation': embedded_solar_generation,
            'embedded_wind_capacity': embedded_wind_capacity,
            'embedded_solar_capacity': embedded_solar_capacity,
            'non_bm_stor': non_bm_stor,
            'pump_storage_pumping': pump_storage_pumping,
            'ifa_flow': ifa_flow,
            'ifa2_flow': ifa2_flow,
            'britned_flow': britned_flow,
            'moyle_flow': moyle_flow,
            'east_west_flow': east_west_flow,
            'nemo_flow': nemo_flow
        }

        # 2. Generate Dummy Data for Initial Lagged and Rolling Features
        df_dummy_history = generate_dummy_lagged_data(start_dt)

        # 3. Iterative 24-Hour (48 Periods) Forecasting
        FORECAST_HORIZON = 48
        predictions = []
        future_dates = []
        current_history = df_dummy_history.copy()

        with st.spinner('Calculating 48 settlement periods...'):
            for step in range(FORECAST_HORIZON):
                current_dt = start_dt + step * timedelta(minutes=30)
                future_dates.append(current_dt)

                settlement_period_current = (current_dt.hour * 2) + (current_dt.minute // 30) + 1
                time_features = generate_time_features(current_dt, uk_holidays)
                lag_roll_features = compute_lagged_and_rolling_features(current_history)

                # Combine all features for this step
                feature_dict = dict(exogenous_dict, **time_features, **lag_roll_features)
                feature_dict['settlement_period'] = settlement_period_current
                
                df_features = pd.DataFrame([feature_dict])[FEATURES_TO_KEEP]

                # Scale and predict
                X_scaled = scaler.transform(df_features)
                pred = model.predict(X_scaled)[0]
                predictions.append(pred)

                # Update history with the new prediction for next iteration
                new_row = pd.DataFrame({'england_wales_demand': [pred]}, index=[current_dt])
                current_history = pd.concat([current_history, new_row]).sort_index()
                if len(current_history) > 336:
                    current_history = current_history.tail(336)

        st.success("✅ 24-Hour Forecast Complete!")

        # 4. Display Results
        st.markdown("---")
        st.subheader("📊 24-Hour Demand Forecast Visualisation")

        # Create DataFrame for plotting
        forecast_df = pd.DataFrame({
            'Datetime': future_dates,
            'Predicted Demand (MW)': predictions
        })

        # Interactive Plot with Aesthetics
        fig = px.line(
            forecast_df, 
            x='Datetime', 
            y='Predicted Demand (MW)',
            title='Predicted England & Wales Demand Over Next 24 Hours',
            labels={'Predicted Demand (MW)': 'Demand (MW)', 'Datetime': 'Time (30-min Intervals)'},
            line_shape='spline'
        )
        # Add styling for better look
        fig.update_traces(line=dict(color='#0083B8', width=3), mode='lines+markers', marker=dict(size=4))
        fig.update_layout(
            xaxis_title='Datetime',
            yaxis_title='Predicted Demand (MW)',
            hovermode="x unified",
            template="plotly_white",
            title_font_size=20
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col_min, col_max = st.columns(2)
        with col_min:
             st.metric(label="Minimum Predicted Demand", value=f"{forecast_df['Predicted Demand (MW)'].min():,.0f} MW")
        with col_max:
             st.metric(label="Maximum Predicted Demand", value=f"{forecast_df['Predicted Demand (MW)'].max():,.0f} MW")

        st.caption("The table below shows the raw predictions for all 48 settlement periods.")
        st.dataframe(forecast_df, use_container_width=True)
    
    st.markdown("---")

    st.write("Adjust the exogenous variables and the starting time in the inputs above, then click **'Generate 24-Hour Forecast'** to test different scenarios.")

