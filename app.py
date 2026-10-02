import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn as nn

# -------------------------------------------------------------
# 1. PAGE SETUP & CLEAN LIGHT THEME (CSS)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Apex Energy AI | Smart Microgrid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    .hero-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .kpi-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-green { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .pill-blue { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. MODEL DEFINITION & ASSET LOADER
# -------------------------------------------------------------
class BiLSTMAttentionEnergyModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, 
                              bidirectional=True, dropout=0.2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.head_total = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.head_submeter = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * att_weights, dim=1)
        norm_context = self.layer_norm(context)
        shared = self.shared_fc(norm_context)
        return self.head_total(shared), self.head_submeter(shared), att_weights

@st.cache_resource
def load_all_artifacts():
    req = ["model.pth", "scaler_X.pkl", "scaler_Y.pkl", "metrics.json", "recent_history.csv"]
    for f in req:
        if not os.path.exists(f):
            return None, None, None, None, None
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_Y = joblib.load("scaler_Y.pkl")
    with open("metrics.json", "r") as fp:
        metrics = json.load(fp)
    recent_history = pd.read_csv("recent_history.csv")
    
    model = BiLSTMAttentionEnergyModel(input_dim=13, hidden_dim=64)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, scaler_X, scaler_Y, metrics, recent_history

model, scaler_X, scaler_Y, metrics, history_df = load_all_artifacts()

if model is None:
    st.error("⚠️ Model artifacts missing! Run 'python train_model.py' in your terminal first.")
    st.stop()

# -------------------------------------------------------------
# 3. LIVE WEATHER FETCHING ENGINE (OPEN-METEO)
# -------------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_live_city_weather(city_name):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=5).json()
        if not geo_resp.get("results"):
            return None, f"City '{city_name}' not found."
        
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        resolved_name = geo_resp["results"][0]["name"] + ", " + geo_resp["results"][0].get("country", "")

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,direct_normal_irradiance&forecast_days=1"
        )
        w_resp = requests.get(weather_url, timeout=5).json()
        hourly = w_resp["hourly"]

        return {
            "resolved_name": resolved_name,
            "temps": hourly["temperature_2m"][:24],
            "humidity": float(np.mean(hourly["relative_humidity_2m"][:24])),
            "solar": hourly["direct_normal_irradiance"][:24]
        }, None
    except Exception as e:
        return None, f"Network timeout/error: {str(e)}"

# -------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.markdown("## 🌐 Input Data Source")
data_mode = st.sidebar.radio("Weather Mode", ["🌍 Live City Telemetry (Open-Meteo)", "🎛️ Manual Scenario Simulator"])

hours_24 = list(range(24))

if data_mode == "🌍 Live City Telemetry (Open-Meteo)":
    st.sidebar.caption("Fetches real-time 24h hourly meteorological forecast:")
    city_choice = st.sidebar.selectbox("Select City", ["Pune", "Mumbai", "Delhi", "Bengaluru", "London", "Tokyo", "New York", "Custom Entry"])
    if city_choice == "Custom Entry":
        city_query = st.sidebar.text_input("Enter City Name", "Nagpur")
    else:
        city_query = city_choice

    live_data, err = fetch_live_city_weather(city_query)
    if live_data:
        st.sidebar.success(f"✓ Connected: {live_data['resolved_name']}")
        hourly_temps = live_data["temps"]
        sim_humidity = live_data["humidity"]
        hourly_solar = live_data["solar"]
        sim_base_temp = float(np.mean(hourly_temps))
    else:
        st.sidebar.warning(f"{err}. Falling back to default baseline.")
        hourly_temps = [28.0 + 6.0 * np.sin(2 * np.pi * (h - 9) / 24.0) for h in hours_24]
        sim_humidity = 55.0
        hourly_solar = [np.clip(np.sin(np.pi * (h - 6) / 12.0) * 850.0, 0, 1000) if (6 <= h <= 18) else 0.0 for h in hours_24]
        sim_base_temp = 28.0
else:
    scenario = st.sidebar.selectbox(
        "Scenario Preset",
        ["Custom Setup", "🔥 Scorching Summer Peak", "⚡ Evening Grid Peak (EV)", "❄️ Mild Winter Afternoon", "🚪 Vacant Home"]
    )
    if scenario == "🔥 Scorching Summer Peak":
        cfg_temp, cfg_hum = 41.5, 45.0
    elif scenario == "⚡ Evening Grid Peak (EV)":
        cfg_temp, cfg_hum = 31.0, 68.0
    elif scenario == "❄️ Mild Winter Afternoon":
        cfg_temp, cfg_hum = 21.0, 40.0
    elif scenario == "🚪 Vacant Home":
        cfg_temp, cfg_hum = 30.0, 50.0
    else:
        cfg_temp, cfg_hum = 33.0, 58.0

    sim_base_temp = st.sidebar.slider("Ambient Base Temp (°C)", 12.0, 48.0, float(cfg_temp), 0.5)
    sim_humidity = st.sidebar.slider("Relative Humidity (%)", 15.0, 95.0, float(cfg_hum), 1.0)
    hourly_temps = [sim_base_temp + 6.0 * np.sin(2 * np.pi * (h - 9) / 24.0) for h in hours_24]
    hourly_solar = [np.clip(np.sin(np.pi * (h - 6) / 12.0) * 850.0, 0, 1000) if (6 <= h <= 18) else 0.0 for h in hours_24]

st.sidebar.markdown("---")
sim_occupancy = st.sidebar.slider("Occupancy Count", 0, 8, 3, 1)
sim_day = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=3)
sim_month = st.sidebar.slider("Month of Year", 1, 12, 6, 1)
is_holiday = 1.0 if st.sidebar.checkbox("Public Holiday Surcharge", value=False) else 0.0

is_weekend = 1.0 if sim_day in ["Saturday", "Sunday"] else 0.0
d_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(sim_day)
d_sin = np.sin(2 * np.pi * d_idx / 7.0)
d_cos = np.cos(2 * np.pi * d_idx / 7.0)
m_sin = np.sin(2 * np.pi * sim_month / 12.0)
m_cos = np.cos(2 * np.pi * sim_month / 12.0)

# -------------------------------------------------------------
# 5. ROBUST 24-HOUR MULTI-HORIZON ROLLOUT INFERENCE
# -------------------------------------------------------------
feature_cols = [
    "temp", "humidity", "solar_irradiance", "occupancy",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "month_sin", "month_cos", "is_weekend", "is_holiday",
    "lag_total_1h"
]

history_buffer = history_df.tail(24)[feature_cols].copy().values
predicted_24h = []
t_start = time.perf_counter()
current_window = history_buffer.copy()

for h in hours_24:
    h_sin = np.sin(2 * np.pi * h / 24.0)
    h_cos = np.cos(2 * np.pi * h / 24.0)
    temp_h = hourly_temps[h]
    solar_h = hourly_solar[h]
    
    occ_h = min(1, sim_occupancy) if h < 6 else (sim_occupancy if (6 <= h < 9 or h >= 17 or is_weekend) else max(0, sim_occupancy - 2))
    lag_val = current_window[-1, 12]

    step_feat = np.array([
        temp_h, sim_humidity, solar_h, occ_h,
        h_sin, h_cos, d_sin, d_cos, m_sin, m_cos,
        is_weekend, is_holiday, lag_val
    ])

    current_window = np.vstack([current_window[1:], step_feat])
    scaled_window = scaler_X.transform(current_window)
    input_t = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        p_tot, p_apps, _ = model(input_t)

    pred_vec = np.hstack([p_tot.numpy(), p_apps.numpy()])
    decoded = scaler_Y.inverse_transform(pred_vec)[0]

    tot_kwh = max(0.12, float(decoded[0]))
    current_window[-1, 12] = tot_kwh

    predicted_24h.append({
        "Hour": h,
        "Time": f"{h:02d}:00",
        "Temperature": round(temp_h, 1),
        "Total_kWh": tot_kwh,
        "AC": max(0.02, float(decoded[1])),
        "Fridge": max(0.04, float(decoded[2])),
        "EV_Charger": max(0.0, float(decoded[3])),
        "Washing_Machine": max(0.0, float(decoded[4])),
        "Induction": max(0.01, float(decoded[5])),
        "Lights_Fans": max(0.02, float(decoded[6])),
        "Solar_PV": max(0.0, float(decoded[7])),
        "Net_Demand": tot_kwh - max(0.0, float(decoded[7]))
    })

rollout_ms = (time.perf_counter() - t_start) * 1000.0
df_forecast_24h = pd.DataFrame(predicted_24h)

daily_total_kwh = df_forecast_24h["Total_kWh"].sum()
daily_solar_kwh = df_forecast_24h["Solar_PV"].sum()

tariff_rates = [12.5 if 18 <= h <= 22 else (8.0 if 6 <= h < 18 else 5.5) for h in hours_24]
daily_cost_inr = sum(max(0.0, row["Net_Demand"]) * rate for row, rate in zip(predicted_24h, tariff_rates))

appliance_sums = {
    "Air Conditioning": df_forecast_24h["AC"].sum(),
    "EV Fast Charger": df_forecast_24h["EV_Charger"].sum(),
    "Kitchen Induction": df_forecast_24h["Induction"].sum(),
    "Washing Machine": df_forecast_24h["Washing_Machine"].sum(),
    "Refrigerator": df_forecast_24h["Fridge"].sum(),
    "Lighting & Fans": df_forecast_24h["Lights_Fans"].sum()
}
top_device = max(appliance_sums, key=appliance_sums.get)

# -------------------------------------------------------------
# 6. HEADER HERO SECTION
# -------------------------------------------------------------
mode_label = f"Live Telemetry ({city_query})" if data_mode.startswith("🌍") else "Synthetic Simulation"
st.markdown(f"""
<div class="hero-box">
    <div>
        <div style="font-size:1.85rem; font-weight:700; color:#0f172a;">
            ⚡ Apex Energy AI <span style="font-size:1.05rem; font-weight:500; color:#64748b;">| Smart Microgrid Intelligence</span>
        </div>
        <div style="color:#64748b; font-size:0.92rem; margin-top:4px;">
            Dual-Level BiLSTM Attention Architecture • Mode: <b>{mode_label}</b>
        </div>
    </div>
    <div style="text-align:right;">
        <span class="pill pill-green">
            <span class="pulse-dot"></span> PyTorch Active
        </span>
        <div style="color:#64748b; font-size:0.8rem; margin-top:6px; font-family:'JetBrains Mono';">
            24h Inference: {rollout_ms:.1f} ms • Params: {sum(p.numel() for p in model.parameters()):,}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 7. TAB NAVIGATION
# -------------------------------------------------------------
tab_fore, tab_arena, tab_nilm, tab_dsm, tab_export = st.tabs([
    "📈 24-Hour Horizon Forecast",
    "🏆 Multi-Model Benchmark Arena",
    "🔌 Appliance Disaggregation (NILM)",
    "⚡ Demand-Side Management (DSM)",
    "📥 Data Export & Defense Summary"
])

# =============================================================
# TAB 1: 24-HOUR HORIZON FORECAST
# =============================================================
with tab_fore:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Projected Daily Demand</div>
        <div class="kpi-number" style="color:#2563eb;">{daily_total_kwh:.1f} <span style="font-size:1rem; color:#94a3b8;">kWh</span></div>
        <div class="kpi-sub" style="color:#2563eb;">Level 1 Household Load</div>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Rooftop Solar PV</div>
        <div class="kpi-number" style="color:#059669;">{daily_solar_kwh:.1f} <span style="font-size:1rem; color:#94a3b8;">kWh</span></div>
        <div class="kpi-sub" style="color:#059669;">Clean Self-Generation</div>
    </div>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Estimated Electricity Cost</div>
        <div class="kpi-number" style="color:#d97706;">₹ {daily_cost_inr:.1f}</div>
        <div class="kpi-sub" style="color:#d97706;">Dynamic ToU Slabs</div>
    </div>
    """, unsafe_allow_html=True)
    c4.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">Primary Load Consumer</div>
        <div class="kpi-number" style="color:#7c3aed;">{top_device.split()[0]}</div>
        <div class="kpi-sub" style="color:#7c3aed;">{appliance_sums[top_device]:.1f} kWh ({(appliance_sums[top_device]/daily_total_kwh)*100:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📊 24-Hour Day-Ahead Load Curve vs. Solar Generation")
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df_forecast_24h["Time"], y=df_forecast_24h["Total_kWh"], mode="lines+markers", name="Household Load", line=dict(color="#2563eb", width=3)))
    fig_h.add_trace(go.Scatter(x=df_forecast_24h["Time"], y=df_forecast_24h["Solar_PV"], mode="lines", name="Solar Generation", line=dict(color="#10b981", width=2.5), fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.12)"))
    fig_h.add_trace(go.Scatter(x=df_forecast_24h["Time"], y=df_forecast_24h["Net_Demand"], mode="lines", name="Net Grid Flow", line=dict(color="#f59e0b", width=2, dash="dash")))
    
    # Grid Parity reference line
    fig_h.add_hline(y=0, line_dash="dot", line_color="#94a3b8", annotation_text="Grid Parity (0 kWh)", annotation_position="bottom right")

    fig_h.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#334155"),
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#e2e8f0", title="Time of Day (24-Hour Timeline)"),
        yaxis=dict(gridcolor="#e2e8f0", title="Power Demand / Generation (kWh)")
    )
    st.plotly_chart(fig_h, use_container_width=True)

# =============================================================
# TAB 2: MULTI-MODEL BENCHMARK ARENA
# =============================================================
with tab_arena:
    st.subheader("🏆 Multi-Model Benchmark Arena (Holdout Validation)")
    st.markdown("""
    This arena provides direct empirical evidence proving that our proposed **BiLSTM + Attention** model outperforms classical ML baselines on temporal time-series records:
    """)

    arena_df = pd.DataFrame(metrics["benchmark_arena"])
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_r2 = px.bar(
            arena_df, x="Model Architecture", y="R² Score", color="Model Architecture",
            color_discrete_sequence=["#94a3b8", "#64748b", "#3b82f6", "#10b981"]
        )
        fig_r2.update_layout(title="Goodness of Fit (R² Score - Higher is Better)", paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320, showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_chart2:
        fig_mae = px.bar(
            arena_df, x="Model Architecture", y="MAE (kWh)", color="Model Architecture",
            color_discrete_sequence=["#f87171", "#fb923c", "#38bdf8", "#34d399"]
        )
        fig_mae.update_layout(title="Mean Absolute Error (MAE - Lower is Better)", paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320, showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    st.write("---")
    st.subheader("📋 Comprehensive Academic Evaluation Matrix")
    st.dataframe(arena_df, use_container_width=True, hide_index=True)

    st.markdown("""
    > **Key Defense Insight:** While Random Forest and Ridge Regression perform adequately on independent static records, they fail to capture sequence momentum. The **BiLSTM + Attention** architecture achieves the highest $R^2$ score and lowest MAE while maintaining physical appliance-sum coherence.
    """)

# =============================================================
# TAB 3: APPLIANCE DISAGGREGATION (NILM)
# =============================================================
with tab_nilm:
    st.subheader("🔌 Level 2 Sub-Metered Load Disaggregation")
    col_p, col_b = st.columns([2, 3])
    with col_p:
        df_p = pd.DataFrame(list(appliance_sums.items()), columns=["Appliance", "Consumption"])
        fig_p = px.pie(df_p, names="Appliance", values="Consumption", hole=0.45, color_discrete_sequence=px.colors.qualitative.Safe)
        fig_p.update_layout(paper_bgcolor="#ffffff", height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True)
    with col_b:
        fig_stk = go.Figure()
        app_cols = {"AC": "#3b82f6", "Fridge": "#10b981", "EV_Charger": "#8b5cf6", "Washing_Machine": "#f59e0b", "Induction": "#ec4899", "Lights_Fans": "#64748b"}
        for k_app, c in app_cols.items():
            fig_stk.add_trace(go.Bar(x=df_forecast_24h["Time"], y=df_forecast_24h[k_app], name=k_app.replace("_", " "), marker_color=c))
        
        # Explicit axis formatting
        fig_stk.update_layout(
            barmode="stack",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="#e2e8f0", title="Hour of Day (24-Hour Horizon)"),
            yaxis=dict(gridcolor="#e2e8f0", title="Hourly Consumption (kWh)")
        )
        st.plotly_chart(fig_stk, use_container_width=True)

# =============================================================
# TAB 4: DEMAND-SIDE MANAGEMENT (DSM)
# =============================================================
with tab_dsm:
    st.subheader("⚡ Demand-Side Management (DSM) Load Shifting Optimizer")
    df_opt = df_forecast_24h.copy()
    ev_peak = df_opt.loc[df_opt["Hour"].isin([18, 19, 20, 21, 22]), "EV_Charger"].sum()
    wash_peak = df_opt.loc[df_opt["Hour"].isin([18, 19, 20, 21, 22]), "Washing_Machine"].sum()

    df_opt.loc[df_opt["Hour"].isin([18, 19, 20, 21, 22]), "EV_Charger"] = 0.0
    df_opt.loc[df_opt["Hour"].isin([18, 19, 20, 21, 22]), "Washing_Machine"] = 0.0
    df_opt.loc[df_opt["Hour"].isin([1, 2]), "EV_Charger"] += ev_peak / 2.0
    df_opt.loc[df_opt["Hour"] == 3, "Washing_Machine"] += wash_peak

    df_opt["Total_Optimized"] = df_opt["AC"] + df_opt["Fridge"] + df_opt["EV_Charger"] + df_opt["Washing_Machine"] + df_opt["Induction"] + df_opt["Lights_Fans"] + 0.11
    df_opt["Net_Optimized"] = df_opt["Total_Optimized"] - df_opt["Solar_PV"]
    opt_cost = sum(max(0.0, row["Net_Optimized"]) * rate for row, rate in zip(df_opt.to_dict('records'), tariff_rates))
    savings = max(0.0, daily_cost_inr - opt_cost)

    c_d1, c_d2, c_d3 = st.columns(3)
    c_d1.metric("Unmanaged Daily Bill", f"₹ {daily_cost_inr:.1f}")
    c_d2.metric("AI-Optimized Bill", f"₹ {opt_cost:.1f}", f"-₹ {savings:.1f} Saved")
    c_d3.metric("Projected Monthly Savings", f"₹ {savings * 30:.0f}", f"{(savings/daily_cost_inr*100):.1f}% Reduction")

    fig_dsm = go.Figure()
    fig_dsm.add_trace(go.Scatter(x=df_forecast_24h["Time"], y=df_forecast_24h["Total_kWh"], mode="lines", name="Unmanaged Demand", line=dict(color="#ef4444", width=2.5, dash="dot")))
    fig_dsm.add_trace(go.Scatter(x=df_opt["Time"], y=df_opt["Total_Optimized"], mode="lines+markers", name="Optimized Schedule (DSM)", line=dict(color="#10b981", width=3)))
    fig_dsm.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_dsm, use_container_width=True)

# =============================================================
# TAB 5: DATA EXPORT & DEFENSE SUMMARY
# =============================================================
with tab_export:
    st.subheader("📥 Export 24-Hour Forecast Dataset")
    csv_bytes = df_forecast_24h.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇️ Download Forecast CSV", data=csv_bytes, file_name="smart_home_energy_forecast_24h.csv", mime="text/csv")
    st.write("")
    st.dataframe(df_forecast_24h, use_container_width=True, hide_index=True)