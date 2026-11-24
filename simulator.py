"""IoT-симулятор датчиків промислового двигуна (варіант 2, практична 9.1)."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SENSORS = ["vibration_amp", "vibration_freq", "bearing_temp",
           "motor_current", "rpm", "acoustic_db", "pressure"]


class IoTSensorSimulator:
    def __init__(self, start_date="2026-01-01", days=60, sampling_interval_minutes=5, seed=42):
        self.rng = np.random.default_rng(seed)
        self.idx = pd.date_range(start_date, periods=days * 24 * 60 // sampling_interval_minutes,
                                 freq=f"{sampling_interval_minutes}min")
        self.df = None

    def _load_factor(self):
        """Навантаження 0..1: змінність (денні години), вихідні, повільний тренд, шум."""
        h = self.idx.hour + self.idx.minute / 60
        daily = 0.5 + 0.4 * np.sin((h - 6) / 24 * 2 * np.pi) * (0.5 + 0.5 * ((h > 6) & (h < 22)))
        weekly = np.where(self.idx.dayofweek >= 5, 0.55, 1.0)
        trend = np.linspace(0, 0.03, len(self.idx))
        return np.clip(daily * weekly + trend + self.rng.normal(0, 0.03, len(self.idx)), 0.05, 1.0)

    def generate_normal_data(self):
        n, r = len(self.idx), self.rng
        load = self._load_factor()
        temp_target = 38 + 27 * load + 1.5 * np.sin(np.arange(n) / (288 * 240) * 2 * np.pi)
        bearing = pd.Series(temp_target).ewm(alpha=0.2).mean().values  # теплова інерція
        rpm = 1450 * (0.92 + 0.08 * load) + r.normal(0, 6, n)
        self.df = pd.DataFrame({
            "timestamp": self.idx,
            "vibration_amp": 1.8 + 1.0 * load + r.normal(0, 0.12, n),       # мм/с
            "vibration_freq": rpm / 60 * (1 + r.normal(0, 0.004, n)),        # Гц
            "bearing_temp": bearing + r.normal(0, 0.6, n),                   # °C
            "motor_current": 18 + 16 * load + r.normal(0, 0.8, n),           # А
            "rpm": rpm,
            "acoustic_db": 62 + 9 * load + r.normal(0, 0.9, n),              # дБ
            "pressure": 5.8 + 0.6 * load + r.normal(0, 0.05, n),             # бар
        })
        self.df["is_anomaly"] = 0
        self.df["anomaly_type"] = "normal"
        return self.df

    def inject_anomalies(self, anomaly_percentage=0.07):
        df, r, n = self.df, self.rng, len(self.df)
        target = int(n * anomaly_percentage)
        marked = np.zeros(n, bool)
        kinds = ["spike", "drift", "level_shift", "outlier", "bearing_fault"]
        i = 0
        while marked.sum() < target:
            kind = kinds[i % len(kinds)]  # по колу: усі типи аномалій представлені рівномірно
            ln = {"spike": r.integers(1, 3), "outlier": 1, "drift": r.integers(12, 30),
                  "level_shift": r.integers(10, 25), "bearing_fault": r.integers(15, 40)}[kind]
            s = int(r.integers(30, n - 100))
            sl = slice(s, s + ln)
            if marked[max(0, s - 5):s + ln + 5].any():
                continue
            k = np.arange(ln)
            if kind == "spike":                      # різкий стрибок кількох датчиків
                df.loc[df.index[sl], "vibration_amp"] += r.uniform(2.0, 3.5)
                df.loc[df.index[sl], "motor_current"] += r.uniform(10, 18)
            elif kind == "drift":                    # повільний дрейф температури підшипника
                df.loc[df.index[sl], "bearing_temp"] += k / ln * r.uniform(10, 18)
            elif kind == "level_shift":              # зсув рівня тиску
                df.loc[df.index[sl], "pressure"] -= r.uniform(0.9, 1.4)
            elif kind == "outlier":                  # одиночний викид одного датчика
                col = SENSORS[r.integers(len(SENSORS))]
                df.loc[df.index[s], col] += r.choice([-1, 1]) * df[col].std() * r.uniform(5, 8)
            else:                                    # знос підшипника: вібрація + шум + нагрів
                g = k / ln
                df.loc[df.index[sl], "vibration_amp"] += 0.8 + 1.8 * g
                df.loc[df.index[sl], "acoustic_db"] += 3 + 9 * g
                df.loc[df.index[sl], "bearing_temp"] += 3 + 9 * g
            marked[sl] = True
            i += 1
            df.loc[df.index[sl], ["is_anomaly", "anomaly_type"]] = [1, kind]
        miss = r.random((n, len(SENSORS))) < 0.004   # пропуски (збої зв'язку)
        for j, c in enumerate(SENSORS):
            df.loc[miss[:, j], c] = np.nan
        return df

    def save_to_csv(self, filename):
        self.df.to_csv(filename, index=False)

    def visualize(self, prefix="figures/sim"):
        d = self.df.iloc[:288 * 7]
        fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        for a, c in zip(ax, ["vibration_amp", "bearing_temp", "motor_current", "pressure"]):
            a.plot(d.timestamp, d[c], lw=.7)
            an = d[d.is_anomaly == 1]
            a.scatter(an.timestamp, an[c], c="red", s=8, label="аномалія")
            a.set_ylabel(c)
        ax[0].legend(); ax[0].set_title("Перший тиждень симуляції (червоні точки - аномалії)")
        plt.tight_layout(); plt.savefig(f"{prefix}_week.png", dpi=130); plt.close()
        self.df.anomaly_type.value_counts().plot.bar(figsize=(7, 4), title="Розподіл типів записів")
        plt.tight_layout(); plt.savefig(f"{prefix}_types.png", dpi=130); plt.close()
        self.df[SENSORS].corr().pipe(lambda c: (plt.figure(figsize=(6, 5)), plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1),
                                                 plt.xticks(range(7), SENSORS, rotation=60), plt.yticks(range(7), SENSORS),
                                                 plt.colorbar(), plt.title("Кореляція датчиків")))
        plt.tight_layout(); plt.savefig(f"{prefix}_corr.png", dpi=130); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--out", default="data/motor_dataset.csv")
    a = p.parse_args()
    sim = IoTSensorSimulator(days=a.days)
    sim.generate_normal_data(); sim.inject_anomalies(0.07)
    sim.save_to_csv(a.out); sim.visualize()
    print(f"Збережено {len(sim.df)} записів, аномалій: {sim.df.is_anomaly.mean():.2%}")
