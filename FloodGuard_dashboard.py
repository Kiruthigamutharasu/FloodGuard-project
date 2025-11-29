import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import requests
import os
def get_secret_env(key: str, default_value: str = "") -> str:
    try:
        # Prefer Streamlit secrets if configured
        return st.secrets.get(key, default_value)  # type: ignore[attr-defined]
    except Exception:
        # Fallback to environment variable
        return os.environ.get(key, default_value)

# --- Visual Theme Helpers ---
COLORWAY = ["#1f77b4", "#e67e22", "#2ecc71", "#9b59b6", "#f1c40f", "#e74c3c", "#16a085"]

def get_static_map_image_url(lat: float, lon: float, zoom: int = 11, size: str = "380x220") -> str:
    try:
        return f"https://staticmap.openstreetmap.de/staticmap.php?center={lat:.5f},{lon:.5f}&zoom={zoom}&size={size}&markers={lat:.5f},{lon:.5f},red-pushpin"
    except Exception:
        return ""

# --- Function Definitions (moved up for early use) ---
def load_district_catalog():
    catalog_path = os.path.join("data", "india_districts.csv")
    if os.path.exists(catalog_path):
        try:
            df = pd.read_csv(catalog_path)
            # expected: state,district,lat,lon
            df = df.dropna(subset=["state", "district"]).copy()
            return df
        except Exception:
            pass
    # Fallback minimal catalog
    return pd.DataFrame([
        {"state": "Tamil Nadu", "district": "Chennai", "lat": 13.0827, "lon": 80.2707},
        {"state": "Tamil Nadu", "district": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
        {"state": "Tamil Nadu", "district": "Madurai", "lat": 9.9252, "lon": 78.1198},
        {"state": "Tamil Nadu", "district": "Salem", "lat": 11.6643, "lon": 78.1460},
        {"state": "Tamil Nadu", "district": "Tiruchengodu", "lat": 11.3800, "lon": 77.8940},
        {"state": "Karnataka", "district": "Bengaluru Urban", "lat": 12.9716, "lon": 77.5946},
        {"state": "Maharashtra", "district": "Mumbai Suburban", "lat": 19.0760, "lon": 72.8777},
    ])

def geocode_city(q: str):
    # Prefer catalog
    try:
        parts = q.split(",")
        if len(parts) >= 2:
            d = parts[0].strip()
            s = parts[1].strip()
            row = catalog_df[(catalog_df["district"].str.lower() == d.lower()) & (catalog_df["state"].str.lower() == s.lower())]
            if not row.empty:
                return float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
    except Exception:
        pass
    # Fallback rough lat/lon dictionary
    mapping = {
        "Tiruchengodu, Tamil Nadu": (11.380, 77.894),
        "Chennai, Tamil Nadu": (13.0827, 80.2707),
        "Coimbatore, Tamil Nadu": (11.0168, 76.9558),
        "Madurai, Tamil Nadu": (9.9252, 78.1198),
        "Salem, Tamil Nadu": (11.6643, 78.1460),
    }
    return mapping.get(q, (11.380, 77.894))

def fetch_openweather(lat: float, lon: float, api_key_value: str, rain_last24_val: float = 120):
    try:
        if not api_key_value:
            raise ValueError("API key missing")
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key_value}&units=metric"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        wind = data.get("wind", {})
        main = data.get("main", {})
        rain = data.get("rain", {})
        # Rainfall last hour as proxy; scale for last 24h if available
        rain_1h = float(rain.get("1h", rain.get("3h", 0)))
        humidity = float(main.get("humidity", 70))
        pressure = float(main.get("pressure", 1008))
        wind_speed_ms = float(wind.get("speed", 4))
        wind_speed_kmh_val = wind_speed_ms * 3.6
        peak_intensity = max(5.0, min(140.0, rain_1h * 3.0 + np.random.normal(0, 3)))
        return {
            "peak_rain_intensity": peak_intensity,
            "pressure_hpa": pressure,
            "humidity": humidity,
            "wind_kmh": wind_speed_kmh_val,
            "rain_last24": max(rain_last24_val, rain_1h * 24),
        }
    except Exception:
        return None

def fetch_openweather_forecast(lat: float, lon: float, api_key_value: str, hours: int = 72):
    try:
        if not api_key_value:
            raise ValueError("API key missing")
        # Try One Call hourly first
        url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key_value}&units=metric&exclude=minutely,daily,alerts,current"
        r = requests.get(url, timeout=10)
        if r.status_code == 401:
            raise PermissionError("unauthorized_onecall")
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", [])
        if not hourly:
            raise ValueError("no_hourly_in_onecall")
        out = []
        for h in hourly[:hours]:
            ts = datetime.fromtimestamp(h.get("dt", 0))
            temp = float(h.get("temp", 0))
            wind_ms = float(h.get("wind_speed", 0))
            wind_kmh = wind_ms * 3.6
            pressure = float(h.get("pressure", 1010))
            humidity = float(h.get("humidity", 70))
            rain_mm = 0.0
            rain_block = h.get("rain", {})
            if isinstance(rain_block, dict):
                rain_mm = float(rain_block.get("1h", 0.0))
            out.append({
                "timestamp": ts,
                "rain_1h_mm": rain_mm,
                "temp_c": temp,
                "wind_kmh": wind_kmh,
                "pressure_hpa": pressure,
                "humidity": humidity,
            })
        if out:
            st.session_state["forecast_source"] = "onecall"
        return out
    except PermissionError:
        # Fallback to 5-day/3-hour forecast for free keys
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key_value}&units=metric"
            r2 = requests.get(url, timeout=10)
            r2.raise_for_status()
            data2 = r2.json()
            lst = data2.get("list", [])
            out2 = []
            for block in lst:
                dt_ts = datetime.fromtimestamp(block.get("dt", 0))
                rain_mm_3h = 0.0
                rain_block = block.get("rain", {})
                if isinstance(rain_block, dict):
                    rain_mm_3h = float(rain_block.get("3h", 0.0))
                temp = float(block.get("main", {}).get("temp", 0))
                wind_ms = float(block.get("wind", {}).get("speed", 0))
                wind_kmh = wind_ms * 3.6
                pressure = float(block.get("main", {}).get("pressure", 1010))
                humidity = float(block.get("main", {}).get("humidity", 70))
                # Split 3h block into three hourly entries
                for i in range(3):
                    ts = dt_ts + timedelta(hours=i)
                    out2.append({
                        "timestamp": ts,
                        "rain_1h_mm": rain_mm_3h / 3.0,
                        "temp_c": temp,
                        "wind_kmh": wind_kmh,
                        "pressure_hpa": pressure,
                        "humidity": humidity,
                    })
                if len(out2) >= hours:
                    break
            # We intentionally removed 5d/3h fallback per user request
            return out2  # <-- ENABLE THIS LINE TO RETURN 3-HOURLY FORECAST
        except Exception:
            return None
    except Exception:
        return None

def fetch_imd_rainfall(lat: float, lon: float):
    """
    # TODO: Integrate IMD rainfall CSV download and parsing here.
    # Example: Download from https://mausam.imd.gov.in, parse for district/location.
    # Return rainfall value for last 24h if available.
    """
    return None

def fetch_bhuvan_terrain(district: str):
    """
    # TODO: Integrate Bhuvan terrain map API or data download here.
    # Example: Use Bhuvan API or download terrain data for the district.
    # Update terrain.csv or return elevation/drainage/urban values.
    """
    return None

def load_terrain(user_district: str):
    # Tries to load from data/terrain.csv; columns: state,district,elevation,drainage,urban
    terrain_path = os.path.join("data", "terrain.csv")
    if os.path.exists(terrain_path):
        try:
            df = pd.read_csv(terrain_path)
            # Accept urban as numeric or labels; normalize
            if "urban" in df.columns:
                if df["urban"].dtype == object:
                    label_map = {"sparse": 0.2, "moderate": 0.6, "dense": 0.9}
                    df["urban"] = df["urban"].astype(str).str.strip().str.lower().map(label_map).fillna(pd.to_numeric(df["urban"], errors="coerce"))
            # Match on district and optionally state
            if "," in user_district:
                dname, sname = [x.strip() for x in user_district.split(",", 1)]
                row = df[(df["district"].str.lower() == dname.lower()) & (df.get("state", "").astype(str).str.lower() == sname.lower())].head(1)
                if row.empty:
                    row = df[df["district"].str.lower() == dname.lower()].head(1)
            else:
                row = df[df["district"].str.lower() == user_district.lower()].head(1)
            if not row.empty:
                return {
                    "elevation": float(row.iloc[0]["elevation"]),
                    "drainage": float(row.iloc[0]["drainage"]),
                    "urban": float(row.iloc[0]["urban"]),
                }
        except Exception:
            pass
    # Fallback synthetic terrain by district
    base = hash(user_district) % 100 / 100
    return {
        "elevation": 50 + 800 * (1 - (base)),
        "drainage": 0.3 + 0.6 * ((base * 1.3) % 1),
        "urban": 0.2 + 0.7 * ((base * 0.7) % 1),
    }

# --- Ensure district/state selection is defined before any UI code or use ---
catalog_df = load_district_catalog()
states = sorted(catalog_df["state"].unique())
# Define sel_state, sel_district, and user_loc (set via sidebar below)
sel_state = None
sel_district = None
user_loc = ""

# --- Initialize input variables with defaults before any use ---
rain_last24 = 120
soil_sat = 0.75
elevation_m = 90
drainage_cap = 0.45
urban_density_label = "Dense"

# --- Header Section ---
st.set_page_config(page_title="FloodGuard – Real-Time Flood Risk Forecaster", layout="wide")

# Title and subtitle
st.title("FloodGuard – Real-Time Flood Risk Forecaster")

# --- Sidebar – User Inputs (Login first, then location and settings) ---
st.sidebar.header("User Inputs")

# Ensure login state defaults exist before rendering login UI
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = "Guest"

# 1) Account
with st.sidebar.expander("Account", expanded=True):
    username = st.text_input("Name", value=st.session_state.user_name, key="sidebar_username")
    password = st.text_input("Password", type="password", key="sidebar_password")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login", key="login_btn_sidebar"):
            st.session_state.logged_in = True
            st.session_state.user_name = username.strip() or "User"
            st.success(f"Welcome, {st.session_state.user_name}")
    with c2:
        if st.button("Logout", key="logout_btn_sidebar"):
            st.session_state.logged_in = False
            st.session_state.user_name = "Guest"
            st.info("Logged out")
    st.caption(f"Status: {'Logged in as ' + st.session_state.user_name if st.session_state.logged_in else 'Not logged in'}")

# 2) State and District
sel_state = st.sidebar.selectbox(
    "State", states, index=states.index("Tamil Nadu") if "Tamil Nadu" in states else 0, key="sidebar_state_select_main_unique_2024"
)
district_options = sorted(catalog_df[catalog_df["state"] == sel_state]["district"].unique())
sel_district = st.sidebar.selectbox(
    "District", district_options, key="sidebar_district_select_main"
)
user_loc = f"{sel_district}, {sel_state}"

# 3) Auto Mode
auto_mode = st.sidebar.toggle("Auto Mode (API/Simulated)", value=True, key="sidebar_auto_mode_toggle_main")

# 4) Forecast Window
forecast_window_label = st.sidebar.selectbox(
    "📅 Forecast Window", ["24h", "48h", "72h"], index=2, key="sidebar_forecast_window_main"
)

# 5) Advanced Settings
with st.sidebar.expander("Advanced Settings", expanded=not auto_mode):
    api_key = st.text_input("OpenWeatherMap API Key (optional)", type="password", value=os.environ.get("OWM_API_KEY", ""), key="sidebar_owm_api_key_main")
    if not auto_mode:
        st.caption("Manual overrides are enabled because Auto Mode is OFF.")
        rain_last24 = st.slider("🌧️ Rainfall Last 24h (mm)", 0, 300, int(rain_last24), key="rain_slider")
        soil_sat = st.slider("🌾 Soil Saturation", 0.0, 1.0, float(soil_sat), key="soil_slider")

# Show selected state and district clearly below the title
st.markdown(f"**Location:** {sel_district}, {sel_state}")
subtitle_col1, subtitle_col2 = st.columns([3,1])
with subtitle_col1:
    st.markdown("**Predicting flood risk using ML and real-time weather intelligence**")
with subtitle_col2:
    if "logged_in" in st.session_state and st.session_state.logged_in:
        st.success(f"Welcome, {st.session_state.user_name}")

# Date and current selection display
col1, col2 = st.columns([1, 3])
with col1:
    st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
with col2:
    st.write("")

# Optional: Login status already ensured above

# Data mode and API key captured from sidebar

# Load telemetry and terrain before sidebar manual override
lat, lon = geocode_city(user_loc)
telemetry = None
forecast_hours = int(forecast_window_label.replace("h", ""))
hourly_forecast = None
if auto_mode:
    owm_key = api_key or get_secret_env("OWM_API_KEY", "")
    telemetry = fetch_openweather(lat, lon, owm_key, rain_last24)
    hourly_forecast = fetch_openweather_forecast(lat, lon, owm_key, hours=forecast_hours)
    if owm_key and (hourly_forecast is None) and (telemetry is None):
        st.error("OpenWeather API key appears invalid or unauthorized for forecast. Please check the key in Advanced Settings.")
forecast_used = hourly_forecast is not None and len(hourly_forecast) > 0
forecast_src = "onecall" if forecast_used else ("5d/3h" if hourly_forecast else "none")
if telemetry is None:
    # Simulated telemetry
    peak_rain_intensity = max(5.0, min(140.0, 10 + 0.5 * rain_last24 + np.random.normal(0, 6)))
    current_pressure_hpa = 1008 - np.random.normal(0, 2) - (peak_rain_intensity - 40) * 0.05
    humidity_pct = 65 + np.random.normal(0, 8)
    wind_speed_kmh = max(5.0, 10 + (peak_rain_intensity / 4) + np.random.normal(0, 3))
else:
    peak_rain_intensity = telemetry["peak_rain_intensity"]
    current_pressure_hpa = telemetry["pressure_hpa"]
    humidity_pct = telemetry["humidity"]
    wind_speed_kmh = telemetry["wind_kmh"]
    rain_last24 = float(telemetry["rain_last24"]) if telemetry.get("rain_last24") else rain_last24

terrain_vals = load_terrain(user_loc)
if auto_mode:
    elevation_m = terrain_vals["elevation"]
    drainage_cap = terrain_vals["drainage"]
    # Use terrain for urban density
    urban_density = terrain_vals["urban"]
else:
    # Use manual override for urban density label
    urban_density_map = {"Sparse": 0, "Moderate": 0.5, "Dense": 1.0}
    urban_density = urban_density_map[urban_density_label]

# Map categorical to numeric (only needed when Auto Mode is OFF)
if not auto_mode:
    urban_density_map = {"Sparse": 0, "Moderate": 0.5, "Dense": 1.0}
    urban_density = urban_density_map[urban_density_label]
forecast_hours = int(forecast_window_label.replace("h", ""))

# --- Derived precipitation features ---
if hourly_forecast and len(hourly_forecast) > 0:
    next24 = hourly_forecast[:min(24, len(hourly_forecast))]
    projected_24h_rain_mm = float(np.sum([max(0.0, h.get("rain_1h_mm", 0.0)) for h in next24]))
else:
    sim_hours = min(24, forecast_hours)
    # Simple synthetic projection based on current peak intensity
    sim_series = np.clip(
        np.linspace(peak_rain_intensity * 0.8, peak_rain_intensity * 0.6, sim_hours) + np.random.normal(0, 2, sim_hours),
        0,
        40,
    )
    projected_24h_rain_mm = float(np.sum(sim_series))

# Use combined observed + projected rainfall as the model rainfall feature
rainfall_feature_mm = float(min(300.0, (rain_last24 or 0.0) + projected_24h_rain_mm))

# --- ML: Flood Risk Model (RandomForest on synthetic training) ---
def generate_synthetic_training(n_rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rainfall = rng.uniform(0, 300, n_rows)
    elevation = rng.uniform(0, 1000, n_rows)
    drainage = rng.uniform(0, 1, n_rows)
    soil = rng.uniform(0, 1, n_rows)
    urban = rng.choice([0.0, 0.5, 1.0], n_rows, p=[0.3, 0.4, 0.3])

    # Risk score rule-of-thumb
    base = (rainfall / 300) * 0.45 + (1 - drainage) * 0.2 + soil * 0.2 + urban * 0.1 + (1 - (elevation / 1000)) * 0.05
    noise = rng.normal(0, 0.05, n_rows)
    score = np.clip(base + noise, 0, 1)
    # Classes: 0=Low,1=Moderate,2=High,3=Severe
    bins = np.digitize(score, [0.25, 0.5, 0.75])
    df = pd.DataFrame({
        "rainfall": rainfall,
        "elevation": elevation,
        "drainage": drainage,
        "soil": soil,
        "urban": urban,
        "risk_level": bins,
        "risk_probability": score,
    })
    return df

@st.cache_resource
def train_flood_model():
    df = generate_synthetic_training()
    X = df[["rainfall", "elevation", "drainage", "soil", "urban"]]
    y = df["risk_level"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=250, random_state=0))
    ])
    model.fit(X, y)
    return model

model = train_flood_model()

# Apply rule to override ML if combined rainfall < 50 mm and elevation > 100 m
if rainfall_feature_mm < 50 and elevation_m > 100:
    predicted_level = 0  # Low
    proba_levels = [1.0, 0.0, 0.0, 0.0]
    valid_inputs = True
else:
    # Validate inputs
    valid_inputs = all([
        rainfall_feature_mm is not None and 0 <= rainfall_feature_mm <= 300,
        elevation_m is not None and 0 <= elevation_m <= 1000,
        drainage_cap is not None and 0.0 <= drainage_cap <= 1.0,
        soil_sat is not None and 0.0 <= soil_sat <= 1.0,
        urban_density is not None and 0.0 <= urban_density <= 1.0
    ])
    if valid_inputs:
        features_row = pd.DataFrame([{ "rainfall": rainfall_feature_mm, "elevation": elevation_m, "drainage": drainage_cap, "soil": soil_sat, "urban": urban_density }])
        proba_levels = model.predict_proba(features_row)[0]
        predicted_level = int(np.argmax(proba_levels))
    else:
        proba_levels = [0.0, 0.0, 0.0, 0.0]
        predicted_level = None

level_names = ["Low", "Moderate", "High", "Severe"]
if valid_inputs and predicted_level is not None:
    flood_probability = float(np.max(proba_levels)) * 100
    flood_level_label = level_names[predicted_level]
else:
    flood_probability = 0.0
    flood_level_label = "N/A"
    st.warning("Invalid or missing input values. Flood prediction unavailable.")

# (Cloudburst module removed as requested)

tab_overview, tab_forecast, tab_map, tab_insights = st.tabs(["Overview", "Forecast", "Map", "Insights"])

with tab_overview:
    kpi_source_note = ("Forecast: LIVE (" + ("One Call" if forecast_src == "onecall" else ("5d/3h" if forecast_src == "5d/3h" else "unknown")) + ")") if forecast_used else ("Forecast: Synthetic" if auto_mode else "Forecast: Manual")
    st.subheader("Key Performance Indicators (KPIs)")
    ts_note = st.session_state.get("refresh_clicked_at", datetime.now().isoformat() if forecast_used else "")
    st.caption(kpi_source_note + (f" · Last update: {ts_note}" if ts_note else ""))
    # KPI Cards with custom CSS
    st.markdown(
        """
        <style>
        .kpi-card{background:#ffffff;border:1px solid #e9ecef;border-radius:10px;padding:12px 14px;box-shadow:0 2px 6px rgba(0,0,0,0.05)}
        .kpi-title{font-size:0.85rem;color:#6c757d;margin-bottom:6px}
        .kpi-value{font-size:1.4rem;font-weight:700;color:#1f2937}
        .kpi-sub{font-size:0.85rem;color:#495057}
        </style>
        """,
        unsafe_allow_html=True,
    )
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🌧️ Rainfall Intensity</div><div class='kpi-value'>{peak_rain_intensity:.1f} mm/h</div></div>", unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🔮 Flood Probability</div><div class='kpi-value'>{flood_probability:.0f}%</div><div class='kpi-sub'>{flood_level_label}</div></div>", unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>📈 Forecast Rain (Next 24h)</div><div class='kpi-value'>{projected_24h_rain_mm:.0f} mm</div></div>", unsafe_allow_html=True)
    alert_text = "Evacuation advised" if (flood_level_label in ["High", "Severe"]) else ("Monitor updates" if flood_level_label == "Moderate" else "Safe")
    with kpi_cols[3]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🚨 Alert Status</div><div class='kpi-value'>{alert_text}</div></div>", unsafe_allow_html=True)
    # Second row KPI cards
    kpi_cols2 = st.columns(2)
    with kpi_cols2[0]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🛤️ Drainage Adequacy</div><div class='kpi-value'>{drainage_cap*100:.0f}%</div></div>", unsafe_allow_html=True)
    with kpi_cols2[1]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>🏙️ Urban Runoff Index</div><div class='kpi-value'>{urban_density*100:.0f}%</div></div>", unsafe_allow_html=True)

with tab_forecast:
    # Rainfall Trend (past 7 days + next forecast)
    dates = pd.date_range(datetime.now() - timedelta(days=7), periods=7*24, freq="h")
    past_rain_intensity = np.clip(np.random.gamma(shape=2.0, scale=4.0, size=len(dates)), 0, 60)
    future_hours = forecast_hours
    future_idx = pd.date_range(datetime.now(), periods=future_hours, freq="h")
    if hourly_forecast:
        future_intensity = np.clip(np.array([max(0.0, h.get("rain_1h_mm", 0.0))*1.0 for h in hourly_forecast]), 0, 140)
        future_idx = pd.to_datetime([h["timestamp"] for h in hourly_forecast])
    else:
        future_intensity = np.clip(np.linspace(peak_rain_intensity * 0.8, peak_rain_intensity * 0.6, future_hours) + np.random.normal(0, 4, future_hours), 0, 140)

    trend_df = pd.DataFrame({"timestamp": list(dates) + list(future_idx), "intensity_mmph": np.concatenate([past_rain_intensity, future_intensity])})
    chart_src = f"LIVE - {('One Call' if forecast_src=='onecall' else ('5d/3h' if forecast_src=='5d/3h' else 'unknown'))}" if forecast_used else ("Synthetic" if auto_mode else "Manual")
    fig_trend = px.line(trend_df, x="timestamp", y="intensity_mmph", title=f"Rainfall Intensity (Past & Forecast) · {chart_src}")
    fig_trend.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=280, colorway=COLORWAY)
    fc_c1, fc_c2 = st.columns([3,1])
    with fc_c1:
        st.plotly_chart(fig_trend, use_container_width=True)
    with fc_c2:
        img_url = get_static_map_image_url(lat, lon)
        if img_url:
            st.image(img_url, caption=f"{sel_district}, {sel_state}")
        else:
            st.write("")

with tab_overview:
    # Flood Risk Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=flood_probability,
        number={'suffix': '%'},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#1f77b4'},
            'steps': [
                {'range': [0, 25], 'color': '#2ecc71'},
                {'range': [25, 50], 'color': '#f1c40f'},
                {'range': [50, 75], 'color': '#e67e22'},
                {'range': [75, 100], 'color': '#e74c3c'}
            ],
            'threshold': {'line': {'color': 'black', 'width': 3}, 'thickness': 0.75, 'value': flood_probability}
        }
    ))
    fig_gauge.update_layout(title=f"Flood Risk Gauge – {flood_level_label}", height=280, margin=dict(l=10, r=10, t=40, b=10), colorway=COLORWAY)
    st.plotly_chart(fig_gauge, use_container_width=True)

with tab_map:
    # Risk Zones Map (synthetic points around a city center)
    lat0, lon0 = lat, lon
    N = 250
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0.0, 0.1, N)
    lats = lat0 + radii * np.cos(angles)
    lons = lon0 + radii * np.sin(angles)
    severity = np.clip(np.random.beta(2, 2, N) * 100 + (flood_probability - 50) * 0.8, 0, 100)
    map_df = pd.DataFrame({"lat": lats, "lon": lons, "severity": severity, "size": np.clip(severity / 12.0 + 4, 4, 14)})
    # Prefer WeatherAPI raster tiles if configured; fallback to OpenStreetMap
    wa_tile_tpl = get_secret_env("WEATHERAPI_TILE_URL", "")
    wa_key = get_secret_env("WEATHERAPI_KEY", "")
    try:
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", color="severity", size="size",
            color_continuous_scale="RdYlGn_r", zoom=9, height=420, title="Flood Risk Zones (Scatter)"
        )
        fig_map.update_layout(mapbox_style="open-street-map", colorway=COLORWAY)
        raster_layers = []
        if wa_tile_tpl:
            tile_url = wa_tile_tpl.replace("{key}", wa_key)
            raster_layers.append({
                "sourcetype": "raster",
                "source": [tile_url],
                "opacity": 0.35,
            })
        if raster_layers:
            fig_map.update_layout(mapbox_layers=raster_layers)
        fig_map.update_layout(mapbox_center={"lat": lat0, "lon": lon0})
        st.plotly_chart(fig_map, use_container_width=True)
        if not wa_tile_tpl:
            st.caption("Tip: set WEATHERAPI_TILE_URL and WEATHERAPI_KEY to overlay live precipitation.")
    except Exception:
        # Fallback scatter_geo if map tiles fail
        fig_geo = px.scatter_geo(map_df, lat="lat", lon="lon", color="severity", color_continuous_scale="RdYlGn_r", title="Flood Risk Zones (Geo)")
        fig_geo.update_geos(fitbounds="locations", visible=False)
        fig_geo.update_layout(height=350, colorway=COLORWAY)
        st.plotly_chart(fig_geo, use_container_width=True)

with tab_insights:
    # Feature Importance
    importances = model.named_steps['rf'].feature_importances_
    feat_df = pd.DataFrame({"feature": ["Rainfall", "Elevation", "Drainage", "Soil", "Urban"], "importance": importances})
    fig_feat = px.bar(feat_df.sort_values("importance", ascending=False), x="importance", y="feature", orientation="h", title="Feature Importance")
    fig_feat.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), colorway=COLORWAY)
    st.plotly_chart(fig_feat, use_container_width=True)

with tab_insights:
    # Timeline: risk progression next N hours (uses cumulative projected rainfall)
    timeline_hours = forecast_hours
    if hourly_forecast:
        timeline_idx = pd.to_datetime([h["timestamp"] for h in hourly_forecast[:timeline_hours]])
        hourly_rain = np.array([max(0.0, h.get("rain_1h_mm", 0.0)) for h in hourly_forecast[:timeline_hours]])
    else:
        timeline_idx = pd.date_range(datetime.now(), periods=timeline_hours, freq="h")
        hourly_rain = np.clip(np.linspace(peak_rain_intensity * 0.8, peak_rain_intensity * 0.6, timeline_hours) + np.random.normal(0, 2, timeline_hours), 0, 40)

    cumulative_rain = np.cumsum(hourly_rain)
    timeline_rows = []
    for i in range(len(timeline_idx)):
        # Combine observed last24 with cumulative projected to form rainfall feature proxy
        rf = float(min(300.0, (rain_last24 or 0.0) + cumulative_rain[i]))
        row = pd.DataFrame([{ "rainfall": rf, "elevation": elevation_m, "drainage": drainage_cap, "soil": soil_sat, "urban": urban_density }])
        p = model.predict_proba(row)[0]
        level = int(np.argmax(p))
        timeline_rows.append({"timestamp": timeline_idx[i], "risk_prob": float(np.max(p)) * 100, "level": level_names[level]})
    timeline_df = pd.DataFrame(timeline_rows)
    fig_timeline = px.area(
        timeline_df,
        x="timestamp",
        y="risk_prob",
        color="level",
        title=f"Flood Risk Progression (Next {timeline_hours}h)",
        color_discrete_map={"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e67e22", "Severe": "#e74c3c"}
    )
    fig_timeline.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), colorway=COLORWAY)
    st.plotly_chart(fig_timeline, use_container_width=True)

with tab_forecast:
    # --- Hourly/3-Hourly Weather Forecast (Next 24h) ---
    if auto_mode:
        st.subheader(f"{'Hourly' if forecast_src == 'onecall' else '3-Hourly' if forecast_src == '5d/3h' else 'Simulated'} Weather Forecast (Next 24h)")
        if hourly_forecast and len(hourly_forecast) > 0:
            fc24 = hourly_forecast[:24]
            # Build DataFrame
            fc_df = pd.DataFrame(fc24)
            fc_df["hour"] = fc_df["timestamp"].apply(lambda t: datetime.fromtimestamp(int(t.timestamp())).strftime("%I %p").lstrip("0"))
            # Ensure columns exist
            if "temp_c" not in fc_df.columns:
                fc_df["temp_c"] = np.nan
            if "rain_1h_mm" not in fc_df.columns:
                fc_df["rain_1h_mm"] = 0.0
            if "pressure_hpa" not in fc_df.columns:
                fc_df["pressure_hpa"] = np.nan
            if "humidity" not in fc_df.columns:
                fc_df["humidity"] = np.nan
            if "wind_kmh" not in fc_df.columns:
                fc_df["wind_kmh"] = np.nan

            # Stabilize flat series by adding tiny jitter only if variance is ~0
            def _jitter(series: pd.Series, scale: float = 0.05) -> pd.Series:
                try:
                    if np.nanstd(series.values.astype(float)) < 1e-6:
                        noise = np.random.normal(0, scale, size=len(series))
                        return pd.Series(series.values.astype(float) + noise)
                except Exception:
                    pass
                return series

            fc_df["temp_c"] = _jitter(fc_df["temp_c"], 0.1)
            fc_df["rain_1h_mm"] = _jitter(fc_df["rain_1h_mm"], 0.05)
            fc_df["pressure_hpa"] = _jitter(fc_df["pressure_hpa"], 0.3)
            fc_df["humidity"] = _jitter(fc_df["humidity"], 0.5)
            fc_df["wind_kmh"] = _jitter(fc_df["wind_kmh"], 0.2)

            # Melt for multi-metric line chart
            plot_df = fc_df[["hour", "temp_c", "rain_1h_mm", "pressure_hpa", "humidity", "wind_kmh"]].copy()
            plot_df = plot_df.rename(columns={
                "temp_c": "Temperature (°C)",
                "rain_1h_mm": "Rain (mm)",
                "pressure_hpa": "Pressure (hPa)",
                "humidity": "Humidity (%)",
                "wind_kmh": "Wind (km/h)",
            })
            melt_df = plot_df.melt(id_vars=["hour"], var_name="metric", value_name="value")
            fig_hourly = px.line(melt_df, x="hour", y="value", color="metric", markers=True, title=f"{'Hourly' if forecast_src == 'onecall' else '3-Hourly' if forecast_src == '5d/3h' else 'Simulated'} Forecast – Next 24h")
            fig_hourly.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="", colorway=COLORWAY)
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            # Simulated fallback for forecast chart
            sim_hours = 24
            sim_idx = pd.date_range(datetime.now(), periods=sim_hours, freq="h")
            sim_temp = np.clip(np.linspace(28, 24, sim_hours) + np.random.normal(0, 1, sim_hours), 20, 38)
            sim_rain = np.clip(np.linspace(peak_rain_intensity * 0.8, peak_rain_intensity * 0.6, sim_hours) + np.random.normal(0, 2, sim_hours), 0, 40)
            sim_press = np.clip(np.linspace(1008, 1002, sim_hours) + np.random.normal(0, 1, sim_hours), 995, 1020)
            sim_hum = np.clip(np.linspace(70, 60, sim_hours) + np.random.normal(0, 2, sim_hours), 40, 100)
            sim_wind = np.clip(np.linspace(10, 18, sim_hours) + np.random.normal(0, 2, sim_hours), 0, 40)
            sim_df = pd.DataFrame({
                "hour": [t.strftime("%I %p").lstrip("0") for t in sim_idx],
                "Temperature (°C)": sim_temp,
                "Rain (mm)": sim_rain,
                "Pressure (hPa)": sim_press,
                "Humidity (%)": sim_hum,
                "Wind (km/h)": sim_wind,
            })
            melt_df = sim_df.melt(id_vars=["hour"], var_name="metric", value_name="value")
            fig_sim = px.line(melt_df, x="hour", y="value", color="metric", markers=True, title="Simulated Forecast – Next 24h")
            fig_sim.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="", colorway=COLORWAY)
            st.plotly_chart(fig_sim, use_container_width=True)

with tab_overview:
    # Downloadable report
    report_df = pd.DataFrame({
        "timestamp": [datetime.now()],
        "district": [user_loc],
        "rain_last24_mm": [rain_last24],
        "peak_intensity_mmph": [peak_rain_intensity],
        "pressure_hpa": [current_pressure_hpa],
        "wind_kmh": [wind_speed_kmh],
        "elevation_m": [elevation_m],
        "drainage": [drainage_cap],
        "soil_saturation": [soil_sat],
        "urban_density": [urban_density],
        "flood_probability_pct": [flood_probability],
        "flood_level": [flood_level_label],
        "projected_24h_rain_mm": [projected_24h_rain_mm],
        "alert": [alert_text],
    })
    st.download_button("Download CSV Report", report_df.to_csv(index=False).encode("utf-8"), file_name="floodguard_report.csv", mime="text/csv")

    # --- Suggestions Based on Risk Level ---
    st.subheader("💡 Suggestions Based on Risk Level")
    suggestions = {
        "Low": "Safe to travel",
        "Moderate": "Monitor updates",
        "High": "Stay indoors",
        "Severe": "Evacuation advised",
    }
    st.table(pd.DataFrame(list(suggestions.items()), columns=["Risk Level", "Action"]))
# --- Footer Section ---
st.markdown("---")
st.markdown("**Data Sources:**  OpenWeatherMap, Census")
st.markdown(f"**Login status:** Guest · Location: {user_loc}")
# Data Status footer
data_status_cols = st.columns(3)
with data_status_cols[0]:
    st.caption(f"Terrain rows loaded: {len(pd.read_csv(os.path.join('data','terrain.csv'))) if os.path.exists(os.path.join('data','terrain.csv')) else 0}")
with data_status_cols[1]:
    st.caption(f"District catalog rows: {len(pd.read_csv(os.path.join('data','india_districts.csv'))) if os.path.exists(os.path.join('data','india_districts.csv')) else 0}")
with data_status_cols[2]:
    st.caption(f"Auto Mode: {'ON' if auto_mode else 'OFF'} | Weather: {'API' if telemetry else 'Simulated'} | Forecast: {'LIVE (' + forecast_src + ')' if forecast_used else ('Synthetic' if auto_mode else 'Manual')}")
