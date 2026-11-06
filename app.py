import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import plotly.express as px
import google.generativeai as genai # For Gemini API integration

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SolarInsight", page_icon="☀️", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('xgboost_solar_model.pkl')

model = load_model()

# SIDEBAR: SYSTEM CONFIGURATION

st.sidebar.title("☀️ SolarInsight")
st.sidebar.caption("Your Solar. Your Power.")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Dashboard", "Energy Planner", "Forecast", "Diagnostics", "About"])

st.sidebar.markdown("---")
st.sidebar.subheader("System Configuration")

# Gemini API Key Input
api_key = st.sidebar.text_input("Gemini API Key (For AI Insights)", type="password")
if api_key:
    genai.configure(api_key=api_key)

# Complete Malaysia Mapping (13 States, 3 FTs)
location = st.sidebar.selectbox("Select Location", [
    "Johor Bahru, Johor", "Alor Setar, Kedah", "Kota Bharu, Kelantan",
    "Kuala Lumpur, FT", "Labuan, FT", "Putrajaya, FT",
    "Malacca City, Melaka", "Seremban, Negeri Sembilan", "Kuantan, Pahang",
    "George Town, Penang", "Ipoh, Perak", "Kangar, Perlis",
    "Kota Kinabalu, Sabah", "Kuching, Sarawak", "Shah Alam, Selangor",
    "Kuala Terengganu, Terengganu"
])


# System Size Slider (1.0 kWp to 10.0 kWp)
system_size = st.sidebar.slider("System Capacity (kWp)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

# TNB Bill Slider 
monthly_bill = st.sidebar.slider("Average TNB Bill Before Solar (RM)", 50, 1000, 250, 10)
    
coords = {
    "Johor Bahru, Johor": (1.4927, 103.7414),
    "Alor Setar, Kedah": (6.1210, 100.3601),
    "Kota Bharu, Kelantan": (6.1254, 102.2381),
    "Kuala Lumpur, FT": (3.1390, 101.6869),
    "Labuan, FT": (5.2767, 115.2417),
    "Putrajaya, FT": (2.9264, 101.6964),
    "Malacca City, Melaka": (2.1896, 102.2501),
    "Seremban, Negeri Sembilan": (2.7258, 101.9424),
    "Kuantan, Pahang": (3.8126, 103.3256),
    "George Town, Penang": (5.4141, 100.3288),
    "Ipoh, Perak": (4.5975, 101.0901),
    "Kangar, Perlis": (6.4414, 100.1986),
    "Kota Kinabalu, Sabah": (5.9804, 116.0735),
    "Kuching, Sarawak": (1.5533, 110.3440),
    "Shah Alam, Selangor": (3.0738, 101.5183),
    "Kuala Terengganu, Terengganu": (5.3117, 103.1324)
}
lat, lon = coords[location]

# API CONNECTION (Pulling 7 days of history)

@st.cache_data(ttl=900) 
def fetch_live_weather(lat: float, lon: float):
    # Added &past_days=7 to retrieve historical data for the bar chart
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,cloud_cover,shortwave_radiation&hourly=temperature_2m,cloud_cover,shortwave_radiation&timezone=Asia%2FSingapore&past_days=7"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        if 'current' in data and 'hourly' in data:
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return None

weather_data = fetch_live_weather(lat, lon)

# System Online Indicator
st.sidebar.markdown("---")
if weather_data:
    st.sidebar.success("🟢 System Online & API Connected")
else:
    st.sidebar.error("🔴 System Offline (Check Connection)")

# MAIN DASHBOARD VIEW

if page == "Dashboard":
    
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.title("Dashboard")
        st.caption(f"Live solar energy metrics for {location} ({system_size} kWp system).")
    
    st.markdown("---")
    
    if weather_data:
        
        # --- Pop-up Notification ---
        st.toast("Live weather and solar data synced successfully!", icon="📡")
        
        # Extract live current conditions
        current_temp = weather_data['current']['temperature_2m']
        current_irr = weather_data['current']['shortwave_radiation']
        current_cloud = weather_data['current']['cloud_cover']
        
        # 1. SHORTER TEXT: Prevents truncation in the 5-column layout
        if current_cloud < 20: condition = "Clear"
        elif current_cloud < 60: condition = "Mixed"
        else: condition = "Overcast"

        # --- DYNAMIC KPI MARKINGS ---
        
        # 2. STREAMLINED DELTAS: Removed the space after the minus sign 
        # so it acts like a clean bullet point, and shortened the descriptions.
        if current_temp > 32:
            temp_delta, temp_color = "High Heat", "inverse"
        else:
            temp_delta, temp_color = "-Optimal Temp", "inverse"
            
        if current_irr > 200:
            irr_delta, irr_color = "Active Sunlight", "normal"
        else:
            irr_delta, irr_color = "-Low Sunlight", "normal"
            
        if current_cloud > 50:
            cloud_delta, cloud_color = f"{current_cloud}% (Heavy)", "inverse"
        else:
            cloud_delta, cloud_color = f"-{current_cloud}% (Light)", "inverse"

        # KPI Summary Cards (Shortened headers to fit cleanly)
        kpi1, kpi_savings, kpi2, kpi3, kpi4 = st.columns(5)
        
        kpi2.metric(label="Temperature 🌡️", value=f"{current_temp} °C", delta=temp_delta, delta_color=temp_color)
        kpi3.metric(label="Irradiance ☀️", value=f"{current_irr} W/m²", delta=irr_delta, delta_color=irr_color)
        kpi4.metric(label="Weather ☁️", value=condition, delta=cloud_delta, delta_color=cloud_color)
        
        st.markdown("---")
        
        # --- EXTRACT SHARED DATA FIRST ---
        # With past_days=7, the first 168 hours (7*24) are history. Today's 24h forecast starts at index 168.
        all_time = weather_data['hourly']['time']
        all_temp = weather_data['hourly']['temperature_2m']
        all_irr = weather_data['hourly']['shortwave_radiation']
        all_cloud = weather_data['hourly']['cloud_cover']
        
        hourly_time = all_time[168:168+24]
        hourly_temp = all_temp[168:168+24]
        hourly_irr = all_irr[168:168+24]
        hourly_cloud = all_cloud[168:168+24]
        
        hist_time = all_time[:168]
        hist_temp = all_temp[:168]
        hist_irr = all_irr[:168]

        # --- RENDER DASHBOARD COLUMNS ---
        chart_col, weather_col = st.columns([2.5, 1])
        
        with chart_col:
            st.subheader("Solar Generation Forecast ⓘ")
            
            forecast_df = pd.DataFrame()
            forecast_df['AMBIENT_TEMPERATURE'] = hourly_temp
            forecast_df['MODULE_TEMPERATURE'] = forecast_df['AMBIENT_TEMPERATURE'] + (np.array(hourly_irr) * 0.025)
            forecast_df['IRRADIATION'] = np.array(hourly_irr) / 1000.0
            
            dt_index = pd.to_datetime(hourly_time)
            hours = dt_index.hour
            months = dt_index.month
            
            forecast_df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            forecast_df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
            forecast_df['month_sin'] = np.sin(2 * np.pi * months / 12.0)
            forecast_df['month_cos'] = np.cos(2 * np.pi * months / 12.0)
            
            forecast_df['generation_lag_1h'] = 0
            forecast_df['generation_lag_24h'] = 0
            
            predictions = model.predict(forecast_df)
            predictions = np.maximum(predictions, 0)
            
            scale_factor = system_size / 25000.0
            scaled_predictions = predictions * scale_factor
            
            total_kwh = scaled_predictions.sum()
            
            #kpi1.metric(label="Predicted Solar Output ⓘ", value=f"{total_kwh:.2f} kWh", delta="24h Cumulative")
            
            total_kwh = scaled_predictions.sum()
        
            # --- NEW: TNB SAVINGS ESTIMATOR MATH ---
            def get_tnb_rate(bill):
                if bill <= 43.60: return 0.218       # First 200 kWh (21.80 sen)
                elif bill <= 77.00: return 0.334     # Next 100 kWh (33.40 sen)
                elif bill <= 231.80: return 0.516    # Next 300 kWh (51.60 sen)
                else: return 0.571                   # >600 kWh (57.10 sen block)
                
            tnb_rate = get_tnb_rate(monthly_bill)
            savings_rm = total_kwh * tnb_rate
            
            # Render the metrics
            kpi1.metric(label="Predicted Solar Output ☀️", value=f"{total_kwh:.2f} kWh", delta="24h Cumulative")
            kpi_savings.metric(label="Est. Savings Today 💰", value=f"RM {savings_rm:.2f}", delta=f"Tier: {tnb_rate*100:.1f} sen/kWh")

            pred_chart_df = pd.DataFrame({
                'Time': pd.to_datetime(hourly_time),
                'Predicted Generation (kW)': scaled_predictions
            })
            
            fig_forecast = px.area(pred_chart_df, x='Time', y='Predicted Generation (kW)', color_discrete_sequence=["#00FFF2D9"])
            fig_forecast.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), 
                height=320,
                xaxis_title="Time of Day",
                yaxis_title="Power Output (kW)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_forecast, width='stretch')
            
        with weather_col:
            st.subheader("24h Weather Trends ⓘ")
            
            df_trends = pd.DataFrame({
                'Time': pd.to_datetime(hourly_time),
                'Temperature (°C)': hourly_temp,
                'Irradiance (W/m²)': hourly_irr,
                'Cloud Cover (%)': hourly_cloud
            })

            def plot_sparkline(df, y_col, color):
                fig = px.line(df, x='Time', y=y_col)
                fig.update_traces(line_color=color, line_width=3)
                fig.update_layout(
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False), 
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=80,
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

            st.write(f"**Temperature (°C):** {current_temp}")
            st.plotly_chart(plot_sparkline(df_trends, 'Temperature (°C)', '#FF4B4B'), width='stretch')
            
            st.write(f"**Solar Irradiance (W/m²):** {current_irr}")
            st.plotly_chart(plot_sparkline(df_trends, 'Irradiance (W/m²)', '#FFA500'), width='stretch')
            
            st.write(f"**Cloud Cover (%):** {current_cloud}")
            st.plotly_chart(plot_sparkline(df_trends, 'Cloud Cover (%)', '#0068C9'), width='stretch')
            
            with st.expander("View 24h Data Table"):
                df_display = df_trends.copy()
                df_display['Time'] = df_display['Time'].dt.strftime('%H:%M')
                st.dataframe(df_display, hide_index=True, width='stretch')

        st.markdown("---")
        st.subheader("Historical Generation (Monsoon vs. Dry Season)")
        st.caption("Select a custom date range to analyze seasonal generation trends.")
        
        # Initialize an empty DataFrame so the CSV export button at the bottom doesn't break
        daily_totals = pd.DataFrame()
        
        # 1. Custom Date Range Selector (Defaults to last 7 days for a clean initial UI)
        default_start = pd.to_datetime('today') - pd.Timedelta(days=7)
        default_end = pd.to_datetime('today') - pd.Timedelta(days=1)
        
        date_selection = st.date_input(
            "Select Date Range (Expand to 3+ months for seasonal trends)", 
            value=(default_start.date(), default_end.date()),
            max_value=default_end.date()
        )
        
        # 2. Wait for user to pick BOTH start and end dates
        if len(date_selection) != 2:
            st.warning("Please select both a Start Date and an End Date.")
        else:
            start_date, end_date = date_selection
            
            # 3. Call the dedicated Open-Meteo Historical Archive API
            archive_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,shortwave_radiation&timezone=auto"
            
            with st.spinner(f"Fetching historical data from {start_date} to {end_date}..."):
                try:
                    arch_res = requests.get(archive_url)
                    arch_res.raise_for_status()
                    arch_data = arch_res.json()
                    
                    if 'hourly' in arch_data:
                        arch_time = arch_data['hourly']['time']
                        arch_temp = arch_data['hourly']['temperature_2m']
                        arch_irr = arch_data['hourly']['shortwave_radiation']
                        
                        # 4. Build DataFrame and run through XGBoost Model
                        arch_df = pd.DataFrame()
                        arch_df['AMBIENT_TEMPERATURE'] = arch_temp
                        arch_df['MODULE_TEMPERATURE'] = arch_df['AMBIENT_TEMPERATURE'] + (np.array(arch_irr) * 0.025)
                        arch_df['IRRADIATION'] = np.array(arch_irr) / 1000.0
                        
                        dt_arch = pd.to_datetime(arch_time)
                        arch_df['hour_sin'] = np.sin(2 * np.pi * dt_arch.hour / 24.0)
                        arch_df['hour_cos'] = np.cos(2 * np.pi * dt_arch.hour / 24.0)
                        arch_df['month_sin'] = np.sin(2 * np.pi * dt_arch.month / 12.0)
                        arch_df['month_cos'] = np.cos(2 * np.pi * dt_arch.month / 12.0)
                        arch_df['generation_lag_1h'] = 0
                        arch_df['generation_lag_24h'] = 0
                        
                        arch_preds = model.predict(arch_df)
                        arch_mask = np.array(arch_irr) / max(max(arch_irr), 1)
                        arch_preds = np.maximum(arch_preds * arch_mask, 0) * (system_size / 25000.0)
                        
                        # 5. Aggregate into Daily Totals for the chart
                        daily_arch_df = pd.DataFrame({
                            'Date': dt_arch.date,
                            'Daily Generation (kWh)': arch_preds
                        })
                        daily_totals = daily_arch_df.groupby('Date')['Daily Generation (kWh)'].sum().reset_index()
                        
                        # Format dates neatly for the chart x-axis
                        daily_totals['Date_Str'] = pd.to_datetime(daily_totals['Date']).dt.strftime('%b %d')
                        
                        # 6. Render Responsive Plotly Chart
                        fig_hist = px.bar(
                            daily_totals, 
                            x='Date_Str', 
                            y='Daily Generation (kWh)',
                            text_auto='.1f',
                            color_discrete_sequence=['#FFA500']
                        )
                        
                        # UI Polish: Hide text numbers if looking at more than 14 days so it doesn't get crowded
                        if len(daily_totals) > 14:
                            fig_hist.update_traces(textposition="none", opacity=0.9)
                        else:
                            fig_hist.update_traces(textfont_size=13, textangle=0, textposition="outside", cliponaxis=False, opacity=0.9)
                            
                        fig_hist.update_layout(
                            xaxis_title="", 
                            yaxis_title="Total Output (kWh)",
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=350,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_hist, width='stretch')
                
                except Exception as e:
                    st.error(f"Failed to load historical data from archive. Error: {e}")
    else:
        st.warning("Failed to load live weather data. Please check your internet connection.")

    st.markdown("---")
    st.subheader("📥 Export Data ⓘ")
    st.caption("Download your generation data for external reporting or analysis.")
    
    # Convert DataFrames to CSV formats
    csv_forecast = pred_chart_df.to_csv(index=False).encode('utf-8')
    csv_history = daily_totals.to_csv(index=False).encode('utf-8')
    
    # Create two columns for the download buttons
    dl_col1, dl_col2, _ = st.columns([1, 1, 2])
    
    with dl_col1:
        st.download_button(
            label="Download 24h Forecast (CSV)",
            data=csv_forecast,
            file_name=f"Solar_Forecast_{location.replace(', ', '_')}.csv",
            mime="text/csv",
            width='stretch'
        )
        
    with dl_col2:
        st.download_button(
            label="Download 7-Day History (CSV)",
            data=csv_history,
            file_name=f"Solar_History_{location.replace(', ', '_')}.csv",
            mime="text/csv",
            width='stretch'
        )

    st.markdown("---")
    st.caption("💡 **Tip:** Sunny and clear days with low cloud cover give the best solar generation.")

elif page == "Energy Planner":
    st.title("⚡ Smart Energy Planner")
    st.caption("Profile your household consumption and compare it against your forecasted generation.")
    
    st.subheader("Appliance Load Calculator")
    st.write("Adjust the sliders to estimate how many hours per day you use these common appliances.")
    
    col1, col2 = st.columns(2)
    with col1:
        ac_hours = st.slider("1.5HP Air-Conditioner (1.2 kW)", 0, 24, 4)
        fridge_hours = st.slider("Refrigerator (0.15 kW)", 0, 24, 24)
        heater_hours = st.slider("Water Heater (3.0 kW)", 0, 24, 1)
        
    with col2:
        washer_hours = st.slider("Washing Machine (0.5 kW)", 0, 24, 1)
        tv_hours = st.slider("TV & Entertainment (0.2 kW)", 0, 24, 4)
        ev_hours = st.slider("EV Charger (7.0 kW)", 0, 24, 0)
        
    # Calculate daily consumption threshold
    daily_consumption = (ac_hours * 1.2) + (fridge_hours * 0.15) + (heater_hours * 3.0) + \
                        (washer_hours * 0.5) + (tv_hours * 0.2) + (ev_hours * 7.0)
                        
    st.metric("Total Estimated Consumption", f"{daily_consumption:.2f} kWh / day")
    
    st.markdown("---")
    st.subheader("Generation vs. Consumption Breakdown")
    
    # Pull the linked data from the Forecast page
    if 'forecast_data' not in st.session_state:
        # EMPTY STATE: Show headers and instructions to guide the user
        st.info("👉 **Action Required:** Please visit the 'Forecast' page first to generate your baseline solar data.")
        
        st.markdown("---")
        st.subheader("💡 Smart Energy Recommendations")
        st.warning("🔒 **Locked:** Awaiting solar forecast data to calculate your optimal appliance schedule.")
        
    else:
        # ACTIVE STATE: Data is loaded, run the math and display the real UI
        plan_df = st.session_state['forecast_data'].copy()
        plan_df['Target Consumption (kWh)'] = round(daily_consumption, 2)
        plan_df['Net Surplus / Deficit (kWh)'] = round(plan_df['Estimated Total (kWh)'] - plan_df['Target Consumption (kWh)'], 2)
        
        st.dataframe(plan_df, hide_index=True, width="stretch")
        
        # --- RULE-BASED ENERGY RECOMMENDATIONS ---
        st.markdown("---")
        st.subheader("💡 Smart Energy Recommendations")
        
        # 1. Math: Find the Best and Worst days mathematically
        best_day_row = plan_df.loc[plan_df['Net Surplus / Deficit (kWh)'].idxmax()]
        worst_day_row = plan_df.loc[plan_df['Net Surplus / Deficit (kWh)'].idxmin()]
        
        try:
            best_date_str = pd.to_datetime(best_day_row['Date']).strftime('%A (%b %d)')
            worst_date_str = pd.to_datetime(worst_day_row['Date']).strftime('%A (%b %d)')
        except:
            best_date_str = str(best_day_row['Date'])
            worst_date_str = str(worst_day_row['Date'])
            
        max_surplus = best_day_row['Net Surplus / Deficit (kWh)']
        max_deficit = worst_day_row['Net Surplus / Deficit (kWh)']
        total_week = plan_df['Net Surplus / Deficit (kWh)'].sum()

        st.write("Based on your custom appliance load and 7-day solar forecast:")

        # 2. Logic: Best Day Recommendation
        if max_surplus > 0:
            st.success(f"**✅ Optimal Usage Day:** {best_date_str}. You have a forecasted surplus of **{max_surplus:.2f} kWh**. Schedule heavy appliances like your washing machine or EV charger during daylight hours on this day to maximize free solar energy.")
        else:
            st.warning(f"**⚠️ Low Generation Week:** Even on your best day, {best_date_str}, you are short by **{abs(max_surplus):.2f} kWh**. Minimize non-essential appliance use during the day.")

        # 3. Logic: Worst Day Recommendation
        if max_deficit < 0:
            st.error(f"**🚨 High Grid-Draw Warning:** On {worst_date_str}, your deficit hits **{abs(max_deficit):.2f} kWh** due to low irradiance. Delay heavy appliance usage on this day to avoid peak grid electricity charges.")
        else:
            st.success(f"**🔋 Excellent Efficiency:** You have zero deficit days this week! Your solar setup covers your estimated load completely.")

        # 4. Logic: Weekly Battery Storage Tip
        if total_week > 10:
            st.info(f"**🔋 Storage Opportunity:** You will generate a net surplus of **{total_week:.2f} kWh** this week. Consider exploring physical battery storage options to capture this excess energy for nighttime use.")
                
elif page == "Diagnostics":
    st.title("Model Diagnostics & Performance")
    st.caption("Validation metrics and internal mechanics of the XGBoost forecasting engine.")
    
    # Top Metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    
    # R-Squared: 0.9962 passes the >= 0.90 target.
    # No minus sign = UP arrow. "normal" color = GREEN.
    m_col1.metric(label="R-Squared (R²)", value="0.9962", delta="Target ≥ 0.90", delta_color="normal")
    
    # RMSE: 8.14% slightly fails the <= 8.0% target (Error is too high).
    # No minus sign = UP arrow. "inverse" color = RED.
    m_col2.metric(label="Normalized RMSE", value="8.14%", delta="Target ≤ 8.0%", delta_color="inverse")
    
    # MAE: 3.85% successfully passes the <= 6.0% target (Error is low).
    # Added a minus sign = DOWN arrow. "inverse" color = GREEN.
    m_col3.metric(label="Normalized MAE", value="3.85%", delta="- Target ≤ 6.0%", delta_color="inverse")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance (Log Scale)")
        st.caption("Note: Irradiation dominates because it is the primary physical source of solar energy.")
        try:
            importance_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            importance_df['Feature'] = importance_df['Feature'].str.replace('_', ' ').str.title()
            
            # Added text labels and changed to a logarithmic x-axis
            fig_imp = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h', 
                color_discrete_sequence=['#FF4B4B'],
                text_auto='.4f'
            )
            fig_imp.update_traces(textposition="outside", cliponaxis=False)
            fig_imp.update_layout(
                height=350, 
                margin=dict(l=0, r=40, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', type='log', title="Importance (Log Scale)"),
                yaxis=dict(showgrid=False, title="")
            )
            st.plotly_chart(fig_imp, width='stretch')
        except Exception as e:
            st.info("Feature importance data unavailable.")

    with col2:
        st.subheader("Hyperparameter Configuration")
        st.caption("Final tuned parameters used for the XGBoost Regressor.")
        
        hyperparams = pd.DataFrame({
            "Parameter": ["Objective Function", "Trees (n_estimators)", "Learning Rate (eta)", "Max Tree Depth", "Subsample Ratio", "Random State"],
            "Value": ["reg:squarederror", "150", "0.05", "6", "0.8", "42"]
        })
        st.dataframe(hyperparams, hide_index=True, width='stretch')
        
        st.subheader("Dataset Profiling")
        st.caption("Training data composition from Kaggle Benchmark.")
        data_profile = pd.DataFrame({
            "Metric": ["Total Observations (15-min)", "Features Engineered", "Target Variable", "Baseline R² (Linear)"],
            "Value": ["3,061", "9", "AC Power (kW)", "0.9935"]
        })
        
elif page == "Forecast":
    st.title("Detailed 7-Day Forecast")
    st.caption("Hour-by-hour and day-ahead generation predictions based on upcoming weather.")

    if weather_data:
        # Extract the 7-day future forecast (from index 168 onwards)
        future_time = weather_data['hourly']['time'][168:]
        future_temp = weather_data['hourly']['temperature_2m'][168:]
        future_irr = weather_data['hourly']['shortwave_radiation'][168:]

        future_df = pd.DataFrame()
        future_df['AMBIENT_TEMPERATURE'] = future_temp
        future_df['MODULE_TEMPERATURE'] = future_df['AMBIENT_TEMPERATURE'] + (np.array(future_irr) * 0.025)
        # CRITICAL FIX: Convert Open-Meteo W/m² to Kaggle kW/m²
        future_df['IRRADIATION'] = np.array(future_irr) / 1000.0

        dt_index_future = pd.to_datetime(future_time)
        future_df['hour_sin'] = np.sin(2 * np.pi * dt_index_future.hour / 24.0)
        future_df['hour_cos'] = np.cos(2 * np.pi * dt_index_future.hour / 24.0)
        future_df['month_sin'] = np.sin(2 * np.pi * dt_index_future.month / 12.0)
        future_df['month_cos'] = np.cos(2 * np.pi * dt_index_future.month / 12.0)
        future_df['generation_lag_1h'] = 0
        future_df['generation_lag_24h'] = 0

        future_preds = model.predict(future_df)
        
        # Apply irradiance mask to force 0 at night
        irr_mask = np.array(future_irr) / max(max(future_irr), 1)
        future_preds = np.maximum(future_preds * irr_mask, 0) * (system_size / 25000.0)

        # --- RENDER WEEK-AHEAD CHART ---
        st.subheader("Week-Ahead Generation Curve")
        week_chart_df = pd.DataFrame({
            'Time': dt_index_future,
            'Predicted Generation (kW)': future_preds
        })
        
        fig_week = px.area(week_chart_df, x='Time', y='Predicted Generation (kW)', color_discrete_sequence=['#00C9A7'])
        fig_week.update_layout(xaxis_title="", yaxis_title="Power Output (kW)", hovermode="x unified", height=400)
        st.plotly_chart(fig_week, width='stretch')

        # --- DAILY SUMMARY TABLE ---
        st.subheader("Daily Generation Breakdown")
        week_chart_df['Date'] = week_chart_df['Time'].dt.date
        daily_future = week_chart_df.groupby('Date')['Predicted Generation (kW)'].sum().reset_index()
        daily_future.columns = ['Date', 'Estimated Total (kWh)']
        daily_future['Estimated Total (kWh)'] = daily_future['Estimated Total (kWh)'].round(2)
        
        st.dataframe(daily_future, width='stretch', hide_index=True)
        
        st.session_state['forecast_data'] = daily_future
        
        # --- AI INSIGHTS GENERATION ---
        st.markdown("---")
        st.subheader("✨ AI Forecast Insights")
        
        if not api_key:
            st.info("Please enter your Gemini API Key in the sidebar to unlock AI summaries.")
        else:
            if st.button("Generate AI Insight"):
                with st.spinner("Analyzing weather and generation trends..."):
                    try:
                        data_string = daily_future.to_string(index=False)
                        prompt = f"""
                        You are an AI assistant in a solar energy dashboard called SolarInsight.
                        Based on the following 7-day predicted solar generation data (in kWh) for a {system_size} kWp system in {location}, provide a short, practical 2-3 sentence summary for the homeowner.
                        Point out the best day for high-energy tasks (like washing machines) and mention any significant drops. Do not give any introduction or robotic phrases, just direct advice.
                        
                        Data:
                        {data_string}
                        """
                        
                        # Explicitly use the modern model required by new API keys
                        model_ai = genai.GenerativeModel('gemini-3.8-flash')
                        response = model_ai.generate_content(
                            prompt,
                            #request_options={"timeout": 50}
                        )
                        
                        st.success(response.text)
                    except Exception as e:
                        st.error(f"Failed to generate insight. Check API key or connection. Error: {e}")                  
        
    else:
        st.warning("Weather data unavailable.")
    
elif page == "About":
    st.title("About SolarInsight")
    st.markdown("---")
    
    # Swapped the order: Solar info is now the default open tab
    tab_solar, tab_project = st.tabs(["Solar in Malaysia", "Project Details"])
    
    with tab_solar:
        st.subheader("Solar Energy Potential in Malaysia")
        st.write("Located near the equator, Malaysia receives abundant sunshine year-round with an average daily solar irradiance of 4.0 - 5.0 kWh/m². While ideal for Photovoltaic (PV) generation, output remains highly subject to cloud cover and the seasonal Northeast/Southwest monsoons.")
        
        st.subheader("Key Policies & Tariffs")
        st.write("- **NEM Rakyat (Net Energy Metering):** A government scheme allowing residential users to export excess solar energy back to the TNB grid on a 1:1 offset basis, maximizing financial returns.")
        st.write("- **Tiered Domestic Tariff:** TNB charges residential users in blocks. The first 200 kWh costs 21.80 sen/kWh, but usage over 600 kWh costs 57.10 sen/kWh. Solar panels drastically reduce bills by eliminating these expensive top-tier blocks.")
        
        st.markdown("---")
        st.subheader("🔗 Official Regulatory Bodies & Resources")
        st.write("For up-to-date regulations, quotas, and grid connection guidelines, refer to the following authorities:")
        
        st.markdown("- **[SEDA Malaysia (Sustainable Energy Development Authority)](https://www.seda.gov.my/)**: Oversees the implementation of the NEM scheme and renewable energy quotas.")
        st.markdown("- **[Tenaga Nasional Berhad (TNB) - Renewable Energy](https://www.tnb.com.my/)**: The national utility company managing grid interconnections and smart meter readings.")
        st.markdown("- **[Suruhanjaya Tenaga (Energy Commission)](https://www.st.gov.my/)**: The regulatory body governing energy safety, licensing, and national grid standards.")

    with tab_project:
        st.subheader("Project Overview")
        st.write("SolarInsight is a forecasting and monitoring platform that predicts short-term solar photovoltaic (PV) generation using an XGBoost ensemble regression model. It provides real-time, data-driven insights for household solar owners and energy planners, mitigating the volatility of renewable energy generation.")
        
        st.subheader("Academic Details")
        st.write("**Prepared By:** Wan Sumayyah binti Wan Mohd Fauzie (22005717)")
        st.write("**Institution:** Universiti Teknologi PETRONAS, Department of Computing")
        st.write("**Supervisor:** Dr. Shakirah bt M Taib")
        st.write("**Project:** Final Year Project II (FYP II)")
        
        st.subheader("System Architecture")
        st.write("- **Machine Learning Core:** XGBoost Regressor (Tuned for R² ≥ 0.90)")
        st.write("- **Live Data Integration:** Open-Meteo API (Weather & Irradiance)")
        st.write("- **Dashboard Framework:** Streamlit (Python)")
        st.write("- **Evaluation Metrics:** RMSE, MAE, R²")
        
        st.caption("Developed for academic evaluation purposes. 2026.")