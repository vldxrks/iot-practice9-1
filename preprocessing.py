"""Попередня обробка та feature engineering (практична 9.1, варіант 2)."""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from simulator import SENSORS

WINDOWS = (3, 6, 12)
MAX_HISTORY = max(WINDOWS)  # кількість записів історії, потрібних для ознак


def explore(df, prefix="figures/eda"):
    """Описова статистика, пропуски, розподіли, box-plot для викидів (IQR)."""
    print(df[SENSORS].describe().round(2).to_string())
    print("Пропуски:\n", df[SENSORS].isna().sum().to_string())
    df[SENSORS].hist(bins=40, figsize=(11, 7)); plt.tight_layout()
    plt.savefig(f"{prefix}_hist.png", dpi=120); plt.close()
    df[SENSORS].plot.box(subplots=True, layout=(2, 4), figsize=(11, 6), sharex=False)
    plt.tight_layout(); plt.savefig(f"{prefix}_box.png", dpi=120); plt.close()
    q1, q3 = df[SENSORS].quantile(.25), df[SENSORS].quantile(.75)
    iqr = q3 - q1
    out = ((df[SENSORS] < q1 - 1.5 * iqr) | (df[SENSORS] > q3 + 1.5 * iqr)).sum()
    print("IQR-викиди (залишаємо: це кандидати в аномалії):\n", out.to_string())


def clean(df):
    """Пропуски: лінійна інтерполяція до 3 підряд (15 хв), далі forward/backward fill."""
    df = df.copy()
    df[SENSORS] = df[SENSORS].interpolate(limit=3, limit_direction="both").ffill().bfill()
    return df


def create_features(df):
    """Часові, лагові, rolling, різницеві та доменні ознаки. Використовується і в train, і в real-time."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["is_business_hours"] = ((df.hour >= 8) & (df.hour < 18) & (df.is_weekend == 0)).astype(int)
    df["month"], df["quarter"] = ts.dt.month, ts.dt.quarter
    for c in SENSORS:
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_diff"] = df[c] - df[f"{c}_lag1"]
        for w in WINDOWS:
            df[f"{c}_rmean{w}"] = df[c].rolling(w).mean()
            df[f"{c}_rstd{w}"] = df[c].rolling(w).std()
    # доменні ознаки: узгодженість датчиків за нормальної роботи
    df["vib_per_current"] = df.vibration_amp / df.motor_current
    df["temp_per_current"] = df.bearing_temp / df.motor_current
    df["acoustic_per_vib"] = df.acoustic_db / df.vibration_amp
    df["power_proxy"] = df.motor_current * df.rpm
    df["freq_rpm_ratio"] = df.vibration_freq / (df.rpm / 60)
    df["pressure_per_current"] = df.pressure / df.motor_current
    df["temp_rmax12"] = df.bearing_temp.rolling(12).max()
    df["temp_rmin12"] = df.bearing_temp.rolling(12).min()
    return df


NB_TARGETS = ["vibration_amp", "bearing_temp", "acoustic_db", "pressure"]
NB_INPUTS = ["motor_current", "motor_current_rmean3", "motor_current_rmean12"]


def fit_normal_model(df_normal):
    """Модель нормальної поведінки: очікуване значення датчика за струмом двигуна (проксі навантаження)."""
    return {t: LinearRegression().fit(df_normal[NB_INPUTS], df_normal[t]) for t in NB_TARGETS}


def add_residuals(df, nb):
    """Залишки (факт - очікуване за нормою) та їх ковзне середнє: чутливі до дрейфу і зсувів."""
    df = df.copy()
    ok = df[NB_INPUTS].notna().all(axis=1)  # у real-time буфері перші рядки ще без rolling
    for t, m in nb.items():
        df[f"resid_{t}"] = np.nan
        df.loc[ok, f"resid_{t}"] = df.loc[ok, t] - m.predict(df.loc[ok, NB_INPUTS])
        df[f"resid_{t}_rmean6"] = df[f"resid_{t}"].rolling(6).mean()
    return df


def feature_columns(df):
    skip = {"timestamp", "is_anomaly", "anomaly_type"}
    return [c for c in df.columns if c not in skip]


def unsupervised_columns(cols):
    """Компактний набір ознак для unsupervised-моделей (сирі датчики, залишки, співвідношення)."""
    keep = list(SENSORS) + [c for c in cols if c.startswith("resid_")] + [
        "vib_per_current", "temp_per_current", "acoustic_per_vib", "freq_rpm_ratio", "pressure_per_current"] + [
        f"{c}_rstd6" for c in ("vibration_amp", "bearing_temp", "motor_current")]
    return [c for c in cols if c in keep]


def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """Хронологічний поділ 70/15/15 (без перемішування)."""
    n = len(df)
    a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def prepare(csv="data/motor_dataset.csv", save=True):
    raw = pd.read_csv(csv, parse_dates=["timestamp"])
    explore(raw)
    df = create_features(clean(raw)).iloc[MAX_HISTORY:].reset_index(drop=True)
    train, _, _ = split_data(df)
    nb = fit_normal_model(train[train.is_anomaly == 0])
    df = add_residuals(df, nb).iloc[6:].reset_index(drop=True)
    cols = feature_columns(df)
    train, val, test = split_data(df)
    scaler = RobustScaler().fit(train.loc[train.is_anomaly == 0, cols])  # лише на нормі train
    if save:
        joblib.dump(scaler, "models/scaler.pkl"); joblib.dump(cols, "models/feature_cols.pkl"); joblib.dump(nb, "models/normal_model.pkl")
        df.to_csv("data/preprocessed.csv", index=False)
    return df, cols, scaler, (train, val, test)


if __name__ == "__main__":
    df, cols, _, (tr, va, te) = prepare()
    print(f"ознак: {len(cols)}; train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print("частка аномалій:", [round(x.is_anomaly.mean(), 3) for x in (tr, va, te)])
    corr = df[cols].corr().abs().where(np.triu(np.ones((len(cols), len(cols))), 1).astype(bool))
    print("пар з |r|>0.95 (мультиколінеарність):", int((corr > 0.95).sum().sum()))
