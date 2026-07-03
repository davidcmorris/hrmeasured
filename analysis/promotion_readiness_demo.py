"""
Promotion-Readiness Index — synthetic demo analysis for the HR Measured case study.

This is illustrative, NOT a real client engagement. Every number in the case
study copy is pulled from the actual output of this script (printed below and
saved to analysis/promotion_readiness_output.txt) rather than invented, so the
write-up stays honest about what the synthetic data actually produced.

Design of the simulation:
- Every employee has one true latent "promotion readiness" trait.
- Behavioral-observation ratings and assessment scores are each noisy,
  imperfectly-reliable indicators of that trait (classic psychometric
  measurement-error setup) — mirroring the site's "validity/reliability"
  framing rather than pretending the observed scores ARE the trait.
- Actual promotion outcomes are generated from the TRUE trait, not the
  noisy observed indicators, so even a well-built model can't hit perfect
  discrimination. That's the point: calibrated uncertainty, not false
  precision.
- Tenure and last performance rating (the "easy" naive proxies most orgs
  default to) are only weakly related to the true trait, by design — to
  demonstrate why measurement design beats convenient proxies.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260115)
N = 450

# One true latent trait: overall promotion readiness.
latent_readiness = rng.normal(0, 1, N)

# Behavioral-observation ratings: structured rubric, 1-5 scale, rated by
# manager against defined behavioral anchors. Noisy indicator of latent trait.
leadership_raw = 0.70 * latent_readiness + rng.normal(0, 0.90, N)
execution_raw = 0.60 * latent_readiness + rng.normal(0, 1.00, N)

def to_five_point(x):
    # Rescale a roughly-N(0,1) variable onto a 1-5 rubric scale.
    scaled = 3 + x * 0.9
    return np.clip(np.round(scaled), 1, 5)

behavioral_leadership = to_five_point(leadership_raw)
behavioral_execution = to_five_point(execution_raw)

# Assessment data: a situational-judgment assessment (reported as a z-score)
# and a structured competency-panel composite (reported 0-100).
judgment_z = 0.65 * latent_readiness + rng.normal(0, 0.90, N)
competency_raw = 0.75 * latent_readiness + rng.normal(0, 0.85, N)
competency_score = np.clip(50 + competency_raw * 14, 0, 100)

# Naive proxies most orgs default to: tenure and last performance rating.
# Deliberately weak / slightly misleading relationship to true readiness —
# long tenure without advancement is not the same thing as readiness.
tenure_years = np.clip(rng.gamma(shape=2.2, scale=1.8, size=N) - 0.12 * latent_readiness, 0.2, 20)
perf_raw = 0.30 * latent_readiness + rng.normal(0, 1.0, N)
performance_rating = np.clip(np.round(3.3 + perf_raw * 0.7), 1, 5)

# Actual promotion outcome within 18 months, generated from the TRUE trait
# (not the noisy observed indicators) at a base rate calibrated to ~25%.
intercept_true = -1.75
logit_true = intercept_true + 1.35 * latent_readiness
p_true = 1 / (1 + np.exp(-logit_true))
promoted = rng.binomial(1, p_true)

df = pd.DataFrame({
    "behavioral_leadership": behavioral_leadership,
    "behavioral_execution": behavioral_execution,
    "judgment_z": judgment_z,
    "competency_score": competency_score,
    "tenure_years": tenure_years,
    "performance_rating": performance_rating,
    "promoted": promoted,
})

# ---------------------------------------------------------------------------
# 2. Train / holdout split
# ---------------------------------------------------------------------------
holdout_mask = rng.random(N) < 0.30
train, holdout = df[~holdout_mask].reset_index(drop=True), df[holdout_mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Models: naive (tenure + performance) vs. full (+ behavioral + assessment)
# ---------------------------------------------------------------------------
def fit_logit(frame, predictors):
    X = sm.add_constant(frame[predictors])
    y = frame["promoted"]
    return sm.Logit(y, X).fit(disp=0)

naive_predictors = ["tenure_years", "performance_rating"]
full_predictors = ["behavioral_leadership", "behavioral_execution", "judgment_z",
                    "competency_score", "tenure_years", "performance_rating"]

naive_model = fit_logit(train, naive_predictors)
full_model = fit_logit(train, full_predictors)

def predict(model, frame, predictors):
    X = sm.add_constant(frame[predictors], has_constant="add")
    X = X[model.params.index]
    return model.predict(X)

naive_holdout_pred = predict(naive_model, holdout, naive_predictors)
full_holdout_pred = predict(full_model, holdout, full_predictors)

# ---------------------------------------------------------------------------
# 4. AUC (rank-based, no sklearn dependency)
# ---------------------------------------------------------------------------
def auc(y_true, y_score):
    y_true = np.asarray(y_true)
    ranks = pd.Series(y_score).rank().values
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

auc_naive = auc(holdout["promoted"], naive_holdout_pred)
auc_full = auc(holdout["promoted"], full_holdout_pred)

# ---------------------------------------------------------------------------
# 5. Calibration (holdout, full model) — 5 bins by predicted probability
# ---------------------------------------------------------------------------
calib = holdout.copy()
calib["pred"] = full_holdout_pred
calib["bin"] = pd.qcut(calib["pred"], 5, duplicates="drop")
calib_table = calib.groupby("bin", observed=True).agg(
    predicted=("pred", "mean"),
    actual=("promoted", "mean"),
    n=("promoted", "size"),
).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Readiness tiers on full holdout population
# ---------------------------------------------------------------------------
calib["tier"] = pd.cut(
    calib["pred"], bins=[-0.01, 0.15, 0.40, 1.0],
    labels=["Not Yet", "Developing", "Ready Now"]
)
tier_table = calib.groupby("tier", observed=True).agg(
    n=("promoted", "size"),
    actual_promotion_rate=("promoted", "mean"),
).reset_index()

# ---------------------------------------------------------------------------
# 7. Write text output
# ---------------------------------------------------------------------------
lines = []
lines.append(f"N = {N}, train = {len(train)}, holdout = {len(holdout)}")
lines.append(f"Base promotion rate (full sample): {df['promoted'].mean():.1%}")
lines.append("")
lines.append("=== Naive model (tenure + performance rating only) ===")
lines.append(str(naive_model.summary2().tables[1]))
lines.append(f"Holdout AUC (naive): {auc_naive:.3f}")
lines.append("")
lines.append("=== Full model (+ behavioral observations + assessment data) ===")
lines.append(str(full_model.summary2().tables[1]))
lines.append(f"Holdout AUC (full): {auc_full:.3f}")
lines.append("")
lines.append("Odds ratios (full model):")
odds = np.exp(full_model.params)
ci = np.exp(full_model.conf_int())
odds_table = pd.DataFrame({"odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1]})
lines.append(str(odds_table))
lines.append("")
lines.append("Calibration table (holdout, full model):")
lines.append(str(calib_table))
lines.append("")
lines.append("Readiness tiers (holdout, full model):")
lines.append(str(tier_table))

output_text = "\n".join(lines)
print(output_text)

with open("analysis/promotion_readiness_output.txt", "w") as f:
    f.write(output_text)

# ---------------------------------------------------------------------------
# 8. Charts — styled to match the site (navy / blue / muted grey)
# ---------------------------------------------------------------------------
NAVY = "#1e3a5f"
BLUE = "#3b82f6"
MUTED = "#6b7280"
BORDER = "#e5e7eb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": "#111827",
    "text.color": "#111827",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

IMG_DIR = "src/img/case-studies"

# Chart 1: AUC comparison
fig, ax = plt.subplots(figsize=(6.5, 4.3))
bars = ax.bar(["Naive proxy\n(tenure + performance)", "Behavioral + assessment\nmodel"],
               [auc_naive, auc_full], color=[MUTED, NAVY], width=0.55)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color=BORDER, linewidth=1, linestyle="--")
ax.set_ylabel("Holdout AUC")
ax.set_title("Discrimination: naive proxy vs. full model", fontsize=14, weight="bold", pad=14)
for bar, val in zip(bars, [auc_naive, auc_full]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
            ha="center", fontsize=12, weight="bold", color="#111827")
fig.tight_layout()
fig.savefig(f"{IMG_DIR}/auc_comparison.svg")
plt.close(fig)

# Chart 2: Odds ratio forest plot (full model, excluding constant)
plot_odds = odds_table.drop(index="const")
label_map = {
    "behavioral_leadership": "Leadership behavior (+1 pt)",
    "behavioral_execution": "Execution behavior (+1 pt)",
    "judgment_z": "Judgment assessment (+1 SD)",
    "competency_score": "Competency panel (+10 pts)",
    "tenure_years": "Tenure (+1 year)",
    "performance_rating": "Performance rating (+1 pt)",
}
# Rescale competency to per-10-point for a more readable odds ratio
comp_row = plot_odds.loc["competency_score"]
plot_odds.loc["competency_score"] = comp_row ** 10  # OR per unit -> per 10 units

plot_odds = plot_odds.rename(index=label_map)
plot_odds = plot_odds.iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 4.5))
y_pos = np.arange(len(plot_odds))
ax.errorbar(plot_odds["odds_ratio"], y_pos,
            xerr=[plot_odds["odds_ratio"] - plot_odds["ci_low"],
                  plot_odds["ci_high"] - plot_odds["odds_ratio"]],
            fmt="o", color=NAVY, ecolor=MUTED, capsize=4, markersize=7)
ax.axvline(1.0, color=BORDER, linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_odds.index, fontsize=11)
ax.set_xlabel("Odds ratio (95% CI), 18-month promotion window")
ax.set_title("What predicts promotion, and how strongly", fontsize=13, weight="bold", pad=14)
fig.subplots_adjust(left=0.30, right=0.96, top=0.86, bottom=0.14)
fig.savefig(f"{IMG_DIR}/odds_ratios.svg")
plt.close(fig)

# Chart 3: Calibration plot
fig, ax = plt.subplots(figsize=(6.5, 5.3))
ax.plot([0, 1], [0, 1], linestyle="--", color=BORDER, linewidth=1)
ax.plot(calib_table["predicted"], calib_table["actual"], marker="o",
        color=NAVY, linewidth=2, markersize=8)
ax.set_xlabel("Predicted promotion probability (model)")
ax.set_ylabel("Actual promotion rate (holdout)")
ax.set_xlim(0, max(0.6, calib_table["predicted"].max() * 1.15))
ax.set_ylim(0, max(0.6, calib_table["actual"].max() * 1.15))
ax.set_title("Calibration: is 30% confidence actually 30%?", fontsize=13, weight="bold", pad=14)
fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.12)
fig.savefig(f"{IMG_DIR}/calibration.svg")
plt.close(fig)

print("\nCharts written to", IMG_DIR)
