"""Uplift modeling on X5 RetailHero: which clients should receive the SMS?

Models (all on CatBoost):
  * response model  — predicts P(purchase) and ignores treatment (common but wrong baseline for targeting)
  * S-learner        — one model with treatment as a feature
  * T-learner        — separate models for treatment and control
  * class transformation — single model on the transformed target Z = Y·W + (1 − Y)·(1 − W)

Evaluation: 5 repeated stratified 70/30 splits; uplift@30%, Qini AUC, uplift AUC on the holdout.
Outputs: reports/results.json, reports/figures/*.png
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, qini_curve, uplift_at_k, uplift_auc_score, uplift_by_percentile
from sklift.models import ClassTransformation, SoloModel, TwoModels

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
K = 0.3
NAVY, ACCENT, LIGHT, GREY = "#1f3a5f", "#e07a1f", "#9fb3c8", "#8a94a3"
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

# --- data --------------------------------------------------------------------------
feat = pd.read_parquet(ROOT / "data" / "features.parquet")
train = pd.read_csv(ROOT / "data" / "uplift_train.csv.gz")
df = train.merge(feat, on="client_id", how="left")
df = pd.get_dummies(df, columns=["gender"], prefix="gender", dtype=int)
X = df.drop(columns=["client_id", "treatment_flg", "target"])
y, w = df["target"].to_numpy(), df["treatment_flg"].to_numpy()
ate = y[w == 1].mean() - y[w == 0].mean()
print(f"clients: {len(df):,}, features: {X.shape[1]}, treatment share: {w.mean():.3f}, ATE: {ate:.4f}")


def cb(seed):
    return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.06, l2_leaf_reg=5,
                              random_seed=seed, thread_count=-1, verbose=0)


def fit_predict(name, seed, Xtr, ytr, wtr, Xte):
    if name == "Response model":
        m = cb(seed).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1], m
    if name == "S-learner":
        m = SoloModel(cb(seed)).fit(Xtr, ytr, wtr)
    elif name == "T-learner":
        m = TwoModels(cb(seed), cb(seed), method="vanilla").fit(Xtr, ytr, wtr)
    elif name == "Class transformation":
        m = ClassTransformation(cb(seed)).fit(Xtr, ytr, wtr)
    return m.predict(Xte), m


MODELS = ["Response model", "S-learner", "T-learner", "Class transformation"]
rows, kept = [], {}
for seed in SEEDS:
    strat = y * 2 + w
    Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X, y, w, test_size=0.3, random_state=seed, stratify=strat)
    for name in MODELS:
        t0 = time.time()
        score, model = fit_predict(name, seed, Xtr, ytr, wtr, Xte)
        rows.append({
            "seed": seed, "model": name,
            "uplift_at_30": uplift_at_k(yte, score, wte, strategy="by_group", k=K),
            "qini_auc": qini_auc_score(yte, score, wte),
            "uplift_auc": uplift_auc_score(yte, score, wte),
            "holdout_ate": yte[wte == 1].mean() - yte[wte == 0].mean(),
        })
        if seed == 0:
            kept[name] = (score, model)
        print(f"seed {seed} {name:22s} uplift@30 = {rows[-1]['uplift_at_30']:.4f}  "
              f"qini = {rows[-1]['qini_auc']:.4f}  ({time.time() - t0:.0f}s)")
    if seed == 0:
        y0, w0, X0 = yte, wte, Xte

res = pd.DataFrame(rows)
summary = (res.groupby("model")[["uplift_at_30", "qini_auc", "uplift_auc"]]
           .agg(["mean", "std"]).reindex(MODELS))
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
res.to_csv(ROOT / "reports" / "cv_results.csv", index=False)
print(summary.round(4).to_string())
best = summary["qini_auc_mean"].drop("Response model").idxmax()
best_score = kept[best][0]

# --- business view on seed-0 holdout -----------------------------------------------
order = np.argsort(-best_score)
ys, ws = y0[order], w0[order]
n = len(ys)
fractions = np.linspace(0.05, 1, 20)


def incremental(frac):
    """Estimated extra purchasers if the top `frac` of clients (by model score) get the SMS."""
    k = int(n * frac)
    yt, wt = ys[:k], ws[:k]
    return (yt[wt == 1].mean() - yt[wt == 0].mean()) * k


inc = np.array([incremental(f) for f in fractions])
inc_all = incremental(1.0)
best_frac = fractions[inc.argmax()]
upl_30 = summary.loc[best, "uplift_at_30_mean"]
pct = uplift_by_percentile(y0, best_score, w0, strategy="by_group", bins=10)
resp_30 = summary.loc["Response model", "uplift_at_30_mean"]

results = {
    "clients_train_file": int(len(df)), "receipt_lines": 45_786_568, "transactions": 8_045_229,
    "features": int(X.shape[1]), "ate_pp": round(100 * ate, 2),
    "best_model": best,
    "summary": summary.round(4).reset_index().to_dict("records"),
    "best_uplift_at_30_pp": round(100 * upl_30, 2),
    "lift_vs_random_targeting": round(upl_30 / ate, 2),
    "response_model_uplift_at_30_pp": round(100 * resp_30, 2),
    "top30_share_of_total_incremental_pct": round(100 * incremental(0.3) / inc_all, 1),
    "optimal_fraction_pct": round(100 * best_frac),
    "optimal_fraction_incremental_vs_all_pct": round(100 * (inc.max() / inc_all - 1), 1),
    "incremental_per_1000_sms_top30": round(1000 * upl_30, 1),
    "incremental_per_1000_sms_all": round(1000 * ate, 1),
    "bottom_decile_uplift_pp": round(100 * pct["uplift"].iloc[-1], 2),
}
(ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

# --- figures ------------------------------------------------------------------------
# 1. Qini curves
fig, ax = plt.subplots(figsize=(7.5, 4.6))
others = iter([LIGHT, ACCENT])
colors = {name: GREY if name == "Response model" else NAVY if name == best else next(others) for name in MODELS}
for name in MODELS:
    xq, yq = qini_curve(y0, kept[name][0], w0)
    ax.plot(xq / n * 100, yq, color=colors[name], lw=2 if name == best else 1.4, label=name)
ax.plot([0, 100], [0, yq[-1]], color="black", ls="--", lw=1, label="Random targeting")
ax.set_xlabel("Clients targeted, % (sorted by model score)")
ax.set_ylabel("Incremental purchasers (Qini)")
ax.set_title("Qini curves on the holdout set", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "qini_curves.png", dpi=150); plt.close(fig)

# 2. Uplift by decile
fig, ax = plt.subplots(figsize=(7.5, 4))
vals = 100 * pct["uplift"].to_numpy()
ax.bar(range(1, 11), vals, color=[NAVY if v >= 0 else ACCENT for v in vals])
ax.axhline(100 * ate, color="black", ls="--", lw=1, label=f"Average effect, {100 * ate:.1f} pp")
ax.set_xticks(range(1, 11)); ax.set_xlabel(f"Decile of predicted uplift ({best})")
ax.set_ylabel("Observed uplift, pp")
ax.set_title("Observed SMS effect by predicted-uplift decile", loc="left", fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "uplift_by_decile.png", dpi=150); plt.close(fig)

# 3. Incremental purchasers vs share targeted
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(100 * fractions, 100 * inc / inc_all, color=NAVY, lw=2, marker="o", ms=3)
ax.axhline(100, color="black", ls="--", lw=1, label="Send to everyone")
ax.axvline(30, color=GREY, ls=":", lw=1)
ax.set_xlabel("Clients receiving the SMS, %"); ax.set_ylabel("Incremental purchasers, % of 'send to all'")
ax.set_title("Targeting a subset keeps most of the effect at a fraction of the cost", loc="left", fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "incremental_vs_targeted.png", dpi=150); plt.close(fig)

# 4. Feature importance of the best model
model = kept[best][1]
est = getattr(model, "estimator", None) or getattr(model, "estimator_trmnt", None)
imp = pd.Series(est.get_feature_importance(), index=est.feature_names_).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh(imp.index, imp.values, color=NAVY)
ax.set_title(f"Top features, {best}", loc="left", fontweight="bold"); ax.set_xlabel("CatBoost importance")
fig.tight_layout(); fig.savefig(FIG / "feature_importance.png", dpi=150); plt.close(fig)
print("figures saved")
