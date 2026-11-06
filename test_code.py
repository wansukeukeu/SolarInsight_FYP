import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SolarInsight", page_icon="☀️️", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("☀️ SolarInsight")
st.sidebar.caption("Your Solar. Your Power.")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Forecast", "Model Performance", "About"])

st.sidebar.markdown("---")
st.sidebar.subheader("System Info")
st.sidebar.write("**Location:**\nKuala Lumpur, Malaysia")
st.sidebar.write("**System Size:**\n5.0 kWp")
st.sidebar.write("**Last Updated:**\n30 May 2026, 11:30 AM")

# --- MAIN DASHBOARD VIEW ---
if page == "Dashboard":
    
    # Header area with Title and Date Selector
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("Dashboard")
        st.caption("Overview of your solar energy forecast and performance.")
    with head_col2:
        selected_date = st.date_input("Select Date") # Defaults to today
    
    st.markdown("---")
    
    # Summary Cards (4 Columns)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Predicted Solar Output ⓘ", value="5.8 kWh", delta="12.4% vs yesterday")
    kpi2.metric(label="Current Temperature ⓘ", value="32.1 °C", delta="Feels like 36.0 °C", delta_color="off")
    kpi3.metric(label="Solar Irradiance ⓘ", value="815 W/m²", delta="High", delta_color="off")
    kpi4.metric(label="Weather Condition ⓘ", value="Sunny", delta="Low cloud cover", delta_color="off")
    
    st.markdown("---")
    
    # Middle Section: Forecast Chart & Weather Trends
    chart_col, weather_col = st.columns([2.5, 1])
    
    with chart_col:
        st.subheader("Solar Generation Forecast ⓘ")
        # Placeholder for the XGBoost output line chart
        # Example: st.line_chart(forecast_dataframe)
        st.info("Line chart rendering area: Predicted vs Typical (Average) over 24h")
        
    with weather_col:
        st.subheader("Weather Conditions ⓘ")
        # Placeholders for mini sparkline charts
        st.write("**Temperature (°C):** 32.1")
        st.line_chart(np.random.randn(10, 1), height=100) # Mock data
        
        st.write("**Solar Irradiance (W/m²):** 815")
        st.line_chart(np.random.randn(10, 1), height=100) # Mock data
        
        st.write("**Cloud Cover (%):** 18")
        st.line_chart(np.random.randn(10, 1), height=100) # Mock data

    st.markdown("---")
    
    # Bottom Section: Historical Bar Chart
    st.subheader("Historical Solar Generation (Last 7 Days) ⓘ")
    # Placeholder for the actual historical generation bar chart
    # Example: st.bar_chart(historical_dataframe)
    st.success("Bar chart rendering area: 7-day historical kWh performance")
    
    # Footer Tip
    st.markdown("---")
    st.caption("💡 **Tip:** Sunny and clear days with low cloud cover give the best solar generation.")
    st.caption("Powered by Streamlit • XGBoost • Python")

# Add conditional blocks for the other pages as you expand the system
elif page == "Forecast": # will be detailed later
    st.title("Detailed Forecast")
    
elif page == "Model Performance": # will be detailed later 
    st.title("Model Performance (RMSE, MAE, R²)")