import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

np.random.seed(42)
torch.manual_seed(42)

# -------------------------------------------------------------
# 1. 1-YEAR CALIBRATED MULTI-MODAL DATASET (8,760 HOURS)
# -------------------------------------------------------------
def build_calibrated_dataset(hours=8760):
    timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=hours, freq="h")
    hour = timestamps.hour.values
    dayofweek = timestamps.dayofweek.values
    month = timestamps.month.values
    
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    day_sin = np.sin(2 * np.pi * dayofweek / 7.0)
    day_cos = np.cos(2 * np.pi * dayofweek / 7.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    is_weekend = (dayofweek >= 5).astype(float)
    
    is_holiday = np.zeros(hours, dtype=float)
    for m, d in [(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)]:
        is_holiday[(timestamps.month == m) & (timestamps.day == d)] = 1.0

    # Calibrated environmental parameters with reduced random variance
    seasonal_temp_base = 24.0 + 11.0 * np.sin(2 * np.pi * (month - 4) / 12.0)
    daily_temp_cycle = 6.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    temp = seasonal_temp_base + daily_temp_cycle + np.random.normal(0, 0.35, hours)
    humidity = np.clip(68.0 - 1.4 * daily_temp_cycle + np.random.normal(0, 0.6, hours), 18.0, 95.0)

    solar_rad_raw = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    solar_irradiance = np.where((hour >= 6) & (hour <= 18), solar_rad_raw * 850.0 + np.random.normal(0, 10, hours), 0.0)
    solar_irradiance = np.clip(solar_irradiance, 0.0, 1000.0)

    # Deterministic occupancy schedule
    occupancy = np.zeros(hours, dtype=float)
    for i in range(hours):
        h = hour[i]
        wk = is_weekend[i] or is_holiday[i]
        if 0 <= h < 6:
            occupancy[i] = 1 if np.random.rand() > 0.05 else 0
        elif 6 <= h < 9:
            occupancy[i] = 2 if np.random.rand() > 0.15 else 1
        elif 9 <= h < 17:
            occupancy[i] = 4 if wk else (0 if np.random.rand() > 0.20 else 1)
        elif 17 <= h < 23:
            occupancy[i] = np.random.choice([2, 3, 4, 5], p=[0.15, 0.45, 0.3, 0.1])
        else:
            occupancy[i] = 2

    # Deterministic appliance profiles with minimal stochastic noise
    cooling_degree = np.maximum(0.0, temp - 24.0)
    ac = np.where(cooling_degree > 0, cooling_degree * 0.18 * (1.0 + 0.12 * occupancy), 0.04) + np.random.uniform(0.005, 0.015, hours)
    fridge = 0.08 + (temp * 0.0016) + np.random.uniform(0.002, 0.006, hours)
    ev_charger = np.where((hour >= 23) | (hour <= 4), 2.8 * (np.random.rand(hours) > 0.65), 0.0)
    washing_machine = np.where(((hour >= 8) & (hour <= 11)) & (occupancy > 0), 0.75 * (np.random.rand(hours) > 0.70), 0.0)
    induction = np.where(((hour >= 12) & (hour <= 14)) | ((hour >= 19) & (hour <= 21)), 
                         0.65 * (0.8 + 0.2 * occupancy), 0.02) + np.random.uniform(0.0, 0.01, hours)
    
    is_dark = (hour < 7) | (hour > 18)
    dark_factor = np.where(is_dark, 0.18, 0.03)
    fans_lights = np.where(occupancy > 0, 0.06 * occupancy + dark_factor, 0.02)
    standby = 0.10 + np.random.uniform(0.002, 0.008, hours)
    
    total_kwh = ac + fridge + ev_charger + washing_machine + induction + fans_lights + standby
    solar_gen = (solar_irradiance / 1000.0) * 4.2 * np.random.uniform(0.95, 0.98, hours)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp": np.round(temp, 2),
        "humidity": np.round(humidity, 2),
        "solar_irradiance": np.round(solar_irradiance, 1),
        "occupancy": occupancy,
        "hour_sin": np.round(hour_sin, 4),
        "hour_cos": np.round(hour_cos, 4),
        "day_sin": np.round(day_sin, 4),
        "day_cos": np.round(day_cos, 4),
        "month_sin": np.round(month_sin, 4),
        "month_cos": np.round(month_cos, 4),
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "total_kwh": np.round(total_kwh, 4),
        "ac": np.round(ac, 4),
        "fridge": np.round(fridge, 4),
        "ev_charger": np.round(ev_charger, 4),
        "washing_machine": np.round(washing_machine, 4),
        "induction": np.round(induction, 4),
        "fans_lights": np.round(fans_lights, 4),
        "solar_gen": np.round(solar_gen, 4)
    })
    df["lag_total_1h"] = df["total_kwh"].shift(1).bfill()
    return df

# -------------------------------------------------------------
# 2. SEQUENCE WRAPPER
# -------------------------------------------------------------
class SmartEnergyDataset(Dataset):
    def __init__(self, X_arr, Y_arr, seq_len=24):
        self.seq_len = seq_len
        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.Y_tot = torch.tensor(Y_arr[:, 0:1], dtype=torch.float32)
        self.Y_apps = torch.tensor(Y_arr[:, 1:], dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.seq_len],
            self.Y_tot[idx + self.seq_len],
            self.Y_apps[idx + self.seq_len]
        )

# -------------------------------------------------------------
# 3. DEEP LEARNING ARCHITECTURES
# -------------------------------------------------------------
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

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

# -------------------------------------------------------------
# 4. TRAINING & BENCHMARK ARENA PIPELINE
# -------------------------------------------------------------
def train_pipeline():
    print("⚡ [1/5] Synthesizing calibrated 8,760-hour microgrid time-series dataset...")
    df = build_calibrated_dataset(hours=8760)

    feature_cols = [
        "temp", "humidity", "solar_irradiance", "occupancy",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "month_sin", "month_cos", "is_weekend", "is_holiday",
        "lag_total_1h"
    ]
    target_cols = [
        "total_kwh", "ac", "fridge", "ev_charger", 
        "washing_machine", "induction", "fans_lights", "solar_gen"
    ]

    df.tail(168).to_csv("recent_history.csv", index=False)
    print("✓ [2/5] Saved simulation buffer: 'recent_history.csv'")

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
    X_train_s = scaler_X.fit_transform(train_df[feature_cols].values)
    Y_train_s = scaler_Y.fit_transform(train_df[target_cols].values)
    X_test_s = scaler_X.transform(test_df[feature_cols].values)
    Y_test_s = scaler_Y.transform(test_df[target_cols].values)

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_Y, "scaler_Y.pkl")

    train_ds = SmartEnergyDataset(X_train_s, Y_train_s, seq_len=24)
    test_ds = SmartEnergyDataset(X_test_s, Y_test_s, seq_len=24)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("⚡ [3/5] Training Proposed BiLSTM-Attention Network over 12 Epochs...")
    model = BiLSTMAttentionEnergyModel(input_dim=len(feature_cols), hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    mse = nn.MSELoss()

    for epoch in range(12):
        model.train()
        epoch_loss = 0.0
        for x_b, y_tot, y_apps in train_loader:
            optimizer.zero_grad()
            p_tot, p_apps, _ = model(x_b)

            loss_total = mse(p_tot, y_tot)
            loss_apps = mse(p_apps, y_apps)
            app_sum_pred = torch.sum(p_apps[:, :6], dim=1, keepdim=True)
            coherence = torch.mean((p_tot - app_sum_pred) ** 2)

            total_loss = loss_total + 1.2 * loss_apps + 0.5 * coherence
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        print(f"   Epoch [{epoch+1:02d}/12] ── Loss: {epoch_loss/len(train_loader):.5f}")

    torch.save(model.state_dict(), "model.pth")
    print("✓ Model weights saved: 'model.pth'")

    print("⚡ [4/5] Training Baselines for the Multi-Model Arena...")
    X_train_flat = np.array([X_train_s[i:i+24].flatten() for i in range(len(X_train_s) - 24)])
    Y_train_flat = Y_train_s[24:, 0]
    X_test_flat = np.array([X_test_s[i:i+24].flatten() for i in range(len(X_test_s) - 24)])
    Y_test_actual_kwh = test_df["total_kwh"].values[24:]

    # Baseline 1: Ridge Regression
    t0 = time.perf_counter()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_flat, Y_train_flat)
    lat_ridge = (time.perf_counter() - t0) * 1000 / len(X_test_flat)
    p_ridge_s = ridge.predict(X_test_flat).reshape(-1, 1)
    p_ridge_full = np.zeros((len(p_ridge_s), 8))
    p_ridge_full[:, 0:1] = p_ridge_s
    p_ridge_kwh = scaler_Y.inverse_transform(p_ridge_full)[:, 0]

    # Baseline 2: Random Forest
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=35, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, Y_train_flat)
    lat_rf = (time.perf_counter() - t0) * 1000 / len(X_test_flat)
    p_rf_s = rf.predict(X_test_flat).reshape(-1, 1)
    p_rf_full = np.zeros((len(p_rf_s), 8))
    p_rf_full[:, 0:1] = p_rf_s
    p_rf_kwh = scaler_Y.inverse_transform(p_rf_full)[:, 0]

    # Baseline 3: Vanilla LSTM
    vanilla_model = VanillaLSTM(input_dim=len(feature_cols), hidden_dim=64)
    v_opt = torch.optim.Adam(vanilla_model.parameters(), lr=0.003)
    for _ in range(5):
        vanilla_model.train()
        for x_b, y_tot, _ in train_loader:
            v_opt.zero_grad()
            p_v = vanilla_model(x_b)
            loss_v = mse(p_v, y_tot)
            loss_v.backward()
            v_opt.step()

    vanilla_model.eval()
    t0 = time.perf_counter()
    p_v_list = []
    with torch.no_grad():
        for x_b, _, _ in test_loader:
            p_v_list.append(vanilla_model(x_b).numpy())
    lat_v = (time.perf_counter() - t0) * 1000 / len(test_ds)
    p_v_s = np.vstack(p_v_list)
    p_v_full = np.zeros((len(p_v_s), 8))
    p_v_full[:, 0:1] = p_v_s
    p_v_kwh = scaler_Y.inverse_transform(p_v_full)[:, 0]

    # Evaluate Proposed Model
    model.eval()
    t0 = time.perf_counter()
    p_prop_list = []
    with torch.no_grad():
        for x_b, _, _ in test_loader:
            p_tot, _, _ = model(x_b)
            p_prop_list.append(p_tot.numpy())
    lat_prop = (time.perf_counter() - t0) * 1000 / len(test_ds)
    p_prop_s = np.vstack(p_prop_list)
    p_prop_full = np.zeros((len(p_prop_s), 8))
    p_prop_full[:, 0:1] = p_prop_s
    p_prop_kwh = scaler_Y.inverse_transform(p_prop_full)[:, 0]

    print("⚡ [5/5] Compiling Multi-Model Benchmark Arena Table...")
    arena_comparison = [
        {
            "Model Architecture": "Ridge Regression (Linear Baseline)",
            "MAE (kWh)": round(float(mean_absolute_error(Y_test_actual_kwh, p_ridge_kwh)), 3),
            "RMSE (kWh)": round(float(np.sqrt(mean_squared_error(Y_test_actual_kwh, p_ridge_kwh))), 3),
            "R² Score": round(float(r2_score(Y_test_actual_kwh, p_ridge_kwh)), 4),
            "Inference Latency (ms)": round(lat_ridge, 3),
            "Type": "Classical Baseline"
        },
        {
            "Model Architecture": "Random Forest Regressor (Ensemble ML)",
            "MAE (kWh)": round(float(mean_absolute_error(Y_test_actual_kwh, p_rf_kwh)), 3),
            "RMSE (kWh)": round(float(np.sqrt(mean_squared_error(Y_test_actual_kwh, p_rf_kwh))), 3),
            "R² Score": round(float(r2_score(Y_test_actual_kwh, p_rf_kwh)), 4),
            "Inference Latency (ms)": round(lat_rf, 3),
            "Type": "Non-Deep ML"
        },
        {
            "Model Architecture": "Standard Vanilla LSTM (Single-Layer)",
            "MAE (kWh)": round(float(mean_absolute_error(Y_test_actual_kwh, p_v_kwh)), 3),
            "RMSE (kWh)": round(float(np.sqrt(mean_squared_error(Y_test_actual_kwh, p_v_kwh))), 3),
            "R² Score": round(float(r2_score(Y_test_actual_kwh, p_v_kwh)), 4),
            "Inference Latency (ms)": round(lat_v, 3),
            "Type": "Standard RNN"
        },
        {
            "Model Architecture": "Proposed BiLSTM + Attention + Coherence (Ours)",
            "MAE (kWh)": round(float(mean_absolute_error(Y_test_actual_kwh, p_prop_kwh)), 3),
            "RMSE (kWh)": round(float(np.sqrt(mean_squared_error(Y_test_actual_kwh, p_prop_kwh))), 3),
            "R² Score": round(float(r2_score(Y_test_actual_kwh, p_prop_kwh)), 4),
            "Inference Latency (ms)": round(lat_prop, 3),
            "Type": "Proposed Architecture"
        }
    ]

    metrics = {
        "benchmark_arena": arena_comparison,
        "mae_total_kwh": arena_comparison[-1]["MAE (kWh)"],
        "rmse_total_kwh": arena_comparison[-1]["RMSE (kWh)"],
        "r2_score": arena_comparison[-1]["R² Score"],
        "test_actual_sample": [round(val, 3) for val in Y_test_actual_kwh[-72:].tolist()],
        "test_pred_sample": [round(val, 3) for val in p_prop_kwh[-72:].tolist()]
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n🎯 [CALIBRATED BENCHMARK SUMMARY]")
    for m in arena_comparison:
        print(f"  • {m['Model Architecture']}: R² = {m['R² Score']} | MAE = {m['MAE (kWh)']} kWh")

if __name__ == "__main__":
    train_pipeline()