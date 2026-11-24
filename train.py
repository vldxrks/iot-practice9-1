"""Навчання та оцінка моделей виявлення аномалій (практична 9.1, варіант 2)."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from autoencoder import Autoencoder
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve)
from preprocessing import prepare, unsupervised_columns

TARGET_RECALL, MAX_FPR = 0.85, 0.05


def pick_threshold(y, score):
    """Поріг на validation: максимальна precision за recall>=0.85 та FPR<0.05; інакше макс. F1."""
    best, best_key = None, None
    for t in np.unique(np.quantile(score, np.linspace(0.5, 0.999, 400))):
        p = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rec, fpr, prec = tp / max(tp + fn, 1), fp / max(fp + tn, 1), tp / max(tp + fp, 1)
        ok = rec >= TARGET_RECALL and fpr < MAX_FPR
        key = (ok, prec if ok else f1_score(y, p, zero_division=0))
        if best_key is None or key > best_key:
            best, best_key = t, key
    return float(best)


def metrics(y, score, thr):
    p = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    k = int(y.sum())
    topk = np.argsort(-score)[:k]
    return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p),
            "f1": f1_score(y, p), "fpr": fp / (fp + tn), "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score), "precision_at_k": float(y[topk].mean()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    df, cols, scaler, (tr, va, te) = prepare(save=True)
    X = {k: scaler.transform(d[cols]) for k, d in zip(("tr", "va", "te"), (tr, va, te))}
    y = {k: d.is_anomaly.values for k, d in zip(("tr", "va", "te"), (tr, va, te))}
    uidx = [cols.index(c) for c in unsupervised_columns(cols)]   # компактний набір для unsupervised
    XU = {k: v[:, uidx] for k, v in X.items()}
    Xn = XU["tr"][y["tr"] == 0]            # "нормальна" поведінка: лише нормальні рядки train
    scores, report, models, t_train = {}, {}, {}, {}

    # 1) Isolation Forest: перебір гіперпараметрів за PR-AUC на validation
    t0, best = time.time(), (-1, None)
    for g in ParameterGrid({"n_estimators": [100, 300], "max_samples": [256, 1024], "max_features": [0.5, 1.0]}):
        m = IsolationForest(random_state=42, n_jobs=-1, **g).fit(Xn)
        s = average_precision_score(y["va"], -m.score_samples(XU["va"]))
        if s > best[0]: best = (s, m, g)
    models["IsolationForest"] = best[1]; t_train["IsolationForest"] = time.time() - t0
    print("IF best:", best[2])
    f_iso = lambda M: -M.score_samples

    # 2) Автоенкодер
    t0, best_ae = time.time(), (-1, None, None)
    for g in ParameterGrid({"hidden": [(16, 6, 16), (32, 12, 32)], "alpha": [1e-3, 1e-2]}):
        ae = Autoencoder(**g).fit(Xn)
        s = average_precision_score(y["va"], ae.score(XU["va"]))
        if s > best_ae[0]: best_ae = (s, ae, g)
    models["Autoencoder"] = best_ae[1]; t_train["Autoencoder"] = time.time() - t0
    print("AE best:", best_ae[2])

    # 3) Logistic Regression (baseline, supervised) та 4) Random Forest (supervised)
    t0 = time.time()
    lr = GridSearchCV(LogisticRegression(max_iter=2000, class_weight="balanced"), {"C": [0.01, 0.1, 1, 10]},
                      cv=TimeSeriesSplit(4), scoring="average_precision", n_jobs=-1).fit(X["tr"], y["tr"])
    models["LogisticRegression"] = lr.best_estimator_; t_train["LogisticRegression"] = time.time() - t0
    t0 = time.time()
    rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
                      {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_split": [2, 10]},
                      cv=TimeSeriesSplit(4), scoring="average_precision", n_jobs=1).fit(X["tr"], y["tr"])
    models["RandomForest"] = rf.best_estimator_; t_train["RandomForest"] = time.time() - t0
    print("LR best:", lr.best_params_, "| RF best:", rf.best_params_)

    def get_scores(name, Xk):
        m = models[name]
        if name in ("IsolationForest", "Autoencoder"): Xk = Xk[:, uidx]
        if name == "IsolationForest": return -m.score_samples(Xk)
        if name == "Autoencoder": return m.score(Xk)
        return m.predict_proba(Xk)[:, 1]

    thresholds, rows = {}, []
    for name in models:
        sv, st = get_scores(name, X["va"]), get_scores(name, X["te"])
        thr = pick_threshold(y["va"], sv); thresholds[name] = thr
        scores[name] = st
        m = metrics(y["te"], st, thr); m["model"] = name; m["train_s"] = round(t_train[name], 1)
        t0 = time.time(); get_scores(name, X["te"][:200]); m["infer_ms_per_row"] = round((time.time() - t0) / 200 * 1000, 3)
        rows.append(m)
    res = pd.DataFrame(rows).set_index("model").round(4)
    res.to_csv("data/model_comparison.csv"); print(res.to_string())

    # збереження: найкраща unsupervised модель (за F1 на test) використовується в inference
    unsup = res.loc[["IsolationForest", "Autoencoder"]]
    best_name = unsup.f1.idxmax()
    joblib.dump(models["IsolationForest"], "models/isolation_forest.pkl")
    joblib.dump(models["Autoencoder"], "models/autoencoder.pkl")
    joblib.dump(models["RandomForest"], "models/random_forest.pkl")
    joblib.dump(models["LogisticRegression"], "models/logreg.pkl")
    json.dump({"thresholds": thresholds, "deployed": best_name, "unsup_cols": unsupervised_columns(cols)}, open("models/config.json", "w"), indent=1)
    print("Для real-time обрано:", best_name)

    # --- Візуалізації ---
    plt.figure(figsize=(7, 5))
    for n, s in scores.items():
        fpr, tpr, _ = roc_curve(y["te"], s); plt.plot(fpr, tpr, label=f"{n} (AUC={res.loc[n,'roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC-криві (test)")
    plt.tight_layout(); plt.savefig("figures/roc.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, n in zip(ax, models):
        r = res.loc[n]; cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        a.imshow(cm, cmap="Blues"); a.set_title(n, fontsize=9)
        for i in range(2):
            for j in range(2): a.text(j, i, int(cm[i, j]), ha="center", va="center")
        a.set_xticks([0, 1], ["норма", "аном."]); a.set_yticks([0, 1], ["норма", "аном."])
    plt.tight_layout(); plt.savefig("figures/confusion.png", dpi=130); plt.close()

    ts = te.timestamp.values[:2 * 288]; s = scores[best_name][:2 * 288]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(ts, te.bearing_temp.values[:2 * 288], lw=.8)
    an = te.is_anomaly.values[:2 * 288] == 1
    ax[0].scatter(ts[an], te.bearing_temp.values[:2 * 288][an], c="red", s=8, label="реальна аномалія")
    ax[0].legend(); ax[0].set_ylabel("bearing_temp")
    ax[1].plot(ts, s, lw=.8); ax[1].axhline(thresholds[best_name], c="r", ls="--", label="поріг"); ax[1].legend()
    ax[1].set_ylabel(f"score ({best_name})"); plt.suptitle("Anomaly score на тестовій вибірці (2 доби)")
    plt.tight_layout(); plt.savefig("figures/scores.png", dpi=130); plt.close()

    imp = pd.Series(models["RandomForest"].feature_importances_, index=cols).nlargest(15)[::-1]
    imp.plot.barh(figsize=(8, 5), title="Важливість ознак (Random Forest, top-15)")
    plt.tight_layout(); plt.savefig("figures/feature_importance.png", dpi=130); plt.close()
    imp[::-1].to_csv("data/feature_importance.csv")

    # аналіз помилок за типами аномалій (для обраної моделі)
    p = (scores[best_name] >= thresholds[best_name]).astype(int)
    te2 = te.assign(pred=p)
    err = te2[te2.is_anomaly == 1].groupby("anomaly_type").pred.agg(["mean", "count"]).rename(columns={"mean": "recall"})
    err.round(3).to_csv("data/recall_by_type.csv"); print(err.round(3))
    print("FP за годинами (топ-5):", te2[(te2.pred == 1) & (te2.is_anomaly == 0)].hour.value_counts().head(5).to_dict())

if __name__ == "__main__":
    main()
