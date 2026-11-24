"""Real-time ML engine: потік IoT -> preprocessing -> модель -> аномалії -> дії (практична 9.1)."""
import json
import logging
import time
from collections import deque
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autoencoder import Autoencoder  # noqa: F401 (потрібен для joblib)
from preprocessing import create_features, add_residuals, split_data
from simulator import SENSORS

HISTORY = 17  # 12 (rolling) + 5 (rolling залишків) записів історії для ознак
logging.basicConfig(filename="alerts.log", level=logging.INFO, format="%(asctime)s %(message)s")


class RealTimeMLEngine:
    def __init__(self, model_dir="models"):
        cfg = json.load(open(f"{model_dir}/config.json"))
        self.name = cfg["deployed"]
        self.threshold = cfg["thresholds"][self.name]
        self.model = joblib.load(f"{model_dir}/{'autoencoder' if self.name == 'Autoencoder' else 'isolation_forest'}.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.cols = joblib.load(f"{model_dir}/feature_cols.pkl")
        self.nb = joblib.load(f"{model_dir}/normal_model.pkl")
        self.uidx = [self.cols.index(c) for c in cfg["unsup_cols"]]
        self.buffer = deque(maxlen=HISTORY)
        self.alerts, self.errors = 0, 0

    def _fill_missing(self, rec):
        """Пропущені значення -> останнє відоме (forward fill), інакше медіана шкали."""
        last = self.buffer[-1] if self.buffer else {}
        for s in SENSORS:
            v = rec.get(s)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                rec[s] = last.get(s, 0.0)
        return rec

    def preprocess(self, rec):
        self.buffer.append(self._fill_missing(dict(rec)))
        if len(self.buffer) < HISTORY:
            return None  # накопичення історії (warm-up)
        df = add_residuals(create_features(pd.DataFrame(self.buffer)), self.nb)
        x = self.scaler.transform(df[self.cols].iloc[[-1]])
        return x[:, self.uidx]

    def predict(self, x):
        err = float(((self.model.net.predict(x) - x) ** 2).mean()) if self.name == "Autoencoder" \
            else float(-self.model.score_samples(x)[0])
        return err, min(err / (2 * self.threshold), 1.0)  # score, "впевненість" 0..1

    def trigger_action(self, ts, score):
        self.alerts += 1
        logging.info("ANOMALY ts=%s score=%.4f threshold=%.4f -> alert + maintenance ticket", ts, score, self.threshold)

    def process_stream(self, stream, realtime_delay=0.0):
        out = []
        for rec in stream:
            t0 = time.perf_counter()
            try:
                x = self.preprocess(rec)
                if x is None:
                    continue
                score, conf = self.predict(x)
                flag = score >= self.threshold
                if flag:
                    self.trigger_action(rec["timestamp"], score)
            except Exception as e:  # помилка одного запису не зупиняє потік
                self.errors += 1; logging.error("processing error: %s", e); continue
            lat = (time.perf_counter() - t0) * 1000
            out.append({"timestamp": rec["timestamp"], "score": score, "flag": int(flag), "latency_ms": lat,
                        "true": rec.get("is_anomaly", 0), "bearing_temp": rec["bearing_temp"]})
            if realtime_delay:
                time.sleep(realtime_delay)
        return pd.DataFrame(out)


def to_stream(df):
    return (r._asdict() for r in df.itertuples(index=False))


def summarize(res, label, wall):
    y, p = res.true.values, res.flag.values
    tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    d = {"scenario": label, "records": len(res), "mean_ms": round(res.latency_ms.mean(), 2),
         "p95_ms": round(res.latency_ms.quantile(.95), 2), "max_ms": round(res.latency_ms.max(), 2),
         "throughput_rps": round(len(res) / wall, 1), "precision": round(tp / max(tp + fp, 1), 3),
         "recall": round(tp / max(tp + fn, 1), 3), "alerts": int(p.sum())}
    print(d); return d


if __name__ == "__main__":
    raw = pd.read_csv("data/motor_dataset.csv", parse_dates=["timestamp"])
    test = raw.iloc[int(len(raw) * 0.85) - HISTORY:].reset_index(drop=True)
    results = []
    # 1) нормальний потік (весь тестовий період, без затримок)
    eng = RealTimeMLEngine(); t0 = time.time()
    res = eng.process_stream(to_stream(test)); results.append(summarize(res, "normal stream", time.time() - t0))
    # 2) burst: 1000 записів підряд без пауз
    eng = RealTimeMLEngine(); t0 = time.time()
    rb = eng.process_stream(to_stream(test.iloc[:1000 + HISTORY])); results.append(summarize(rb, "burst 1000", time.time() - t0))
    # 3) 1 Гц: 30 записів з паузою 1 с
    eng = RealTimeMLEngine(); t0 = time.time()
    r1 = eng.process_stream(to_stream(test.iloc[:HISTORY + 30]), realtime_delay=1.0); results.append(summarize(r1, "1 Hz (30 s)", time.time() - t0))
    # 4) додаткові пропуски: 15% значень -> NaN
    noisy = test.copy(); rng = np.random.default_rng(0)
    for s in SENSORS:
        noisy.loc[rng.random(len(noisy)) < 0.15, s] = np.nan
    eng = RealTimeMLEngine(); t0 = time.time()
    rn = eng.process_stream(to_stream(noisy)); results.append(summarize(rn, "15% missing", time.time() - t0))
    json.dump(results, open("data/realtime_benchmark.json", "w"), indent=1, ensure_ascii=False)
    res.to_csv("data/realtime_results.csv", index=False)

    seg = res.iloc[:600]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(seg.timestamp, seg.bearing_temp, lw=.8); ax[0].set_ylabel("bearing_temp")
    ax[0].scatter(seg.timestamp[seg.true == 1], seg.bearing_temp[seg.true == 1], c="red", s=9, label="реальна аномалія"); ax[0].legend()
    ax[1].plot(seg.timestamp, seg.score, lw=.8); ax[1].axhline(eng.threshold, c="r", ls="--", label="поріг"); ax[1].set_yscale("log")
    ax[1].set_ylabel("anomaly score"); ax[1].legend()
    ax[2].step(seg.timestamp, seg.flag, label="модель"); ax[2].step(seg.timestamp, seg.true, alpha=.6, label="істина"); ax[2].legend()
    ax[2].set_ylabel("аномалія 0/1"); plt.suptitle("Real-time обробка потоку: прогноз vs реальність")
    plt.tight_layout(); plt.savefig("figures/realtime.png", dpi=130); plt.close()
    plt.figure(figsize=(7, 4)); plt.hist(res.latency_ms, bins=50); plt.xlabel("латентність, мс"); plt.title("Розподіл латентності inference")
    plt.tight_layout(); plt.savefig("figures/latency.png", dpi=130); plt.close()
