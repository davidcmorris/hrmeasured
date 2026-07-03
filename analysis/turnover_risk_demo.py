"""
Turnover Risk Score — synthetic demo analysis for the HR Measured case study
on workforce prediction.

Illustrative, NOT a real client engagement. Every number in the case study
copy comes from the actual output of this script (printed below and saved to
analysis/turnover_risk_output.txt).

Narrative angle (distinct from the other two case studies):
- Case 1 (promotion readiness) showed that "easy" administrative data
  (tenure, performance rating) is weak, and the fix is better inputs.
- Case 2 (high potential) was entirely about measurement reliability.
- This one asks a different question: most organizations already have ONE
  decent predictor of turnover (engagement survey score) and stop there.
  The point here is that turnover is not just an individual attitude problem
  — pay competitiveness, promotion stalling, and peer attrition ("turnover
  contagion": Felps et al., 2009, Academy of Management Journal) all add
  real, independent signal beyond what a single engagement score captures.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260710)
N = 600

# ---------------------------------------------------------------------------
# 1. Predictors
# ---------------------------------------------------------------------------
engagement = np.clip(np.round(rng.normal(3.4, 0.7, N) * 2) / 2, 1, 5)  # survey composite, 0.5 increments
comp_ratio = np.clip(rng.normal(1.00, 0.12, N), 0.65, 1.35)  # pay vs. market band midpoint
months_since_promotion = np.clip(rng.gamma(shape=2.0, scale=14, size=N), 1, 96)
peer_attrition_rate = np.clip(rng.beta(1.4, 7.0, N), 0, 0.8)  # trailing 6mo attrition on immediate team

# ---------------------------------------------------------------------------
# 2. True turnover process (directly from observed predictors this time —
#    no measurement-error layer; the lesson here is about which predictors
#    matter, not about rater or instrument noise)
# ---------------------------------------------------------------------------
def zscore(x):
    return (x - x.mean()) / x.std()

log_stall = np.log1p(months_since_promotion)

logit_true = (
    -1.85
    - 0.85 * zscore(engagement)
    - 0.55 * zscore(comp_ratio)
    + 0.45 * zscore(log_stall)
    + 0.65 * zscore(peer_attrition_rate)
)
p_true = 1 / (1 + np.exp(-logit_true))
left_within_6mo = rng.binomial(1, p_true)

df = pd.DataFrame({
    "engagement": engagement,
    "comp_ratio": comp_ratio,
    "months_since_promotion": months_since_promotion,
    "peer_attrition_rate": peer_attrition_rate,
    "left_within_6mo": left_within_6mo,
})

# ---------------------------------------------------------------------------
# 3. Train / holdout split
# ---------------------------------------------------------------------------
holdout_mask = rng.random(N) < 0.30
train, holdout = df[~holdout_mask].reset_index(drop=True), df[holdout_mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Models: naive (engagement only) vs. full (+ comp, stalling, peer attrition)
# ---------------------------------------------------------------------------
def fit_logit(frame, predictors):
    X = sm.add_constant(frame[predictors])
    y = frame["left_within_6mo"]
    return sm.Logit(y, X).fit(disp=0)

naive_predictors = ["engagement"]
full_predictors = ["engagement", "comp_ratio", "months_since_promotion", "peer_attrition_rate"]

naive_model = fit_logit(train, naive_predictors)
full_model = fit_logit(train, full_predictors)

def predict(model, frame, predictors):
    X = sm.add_constant(frame[predictors], has_constant="add")
    X = X[model.params.index]
    return model.predict(X)

naive_holdout_pred = predict(naive_model, holdout, naive_predictors)
full_holdout_pred = predict(full_model, holdout, full_predictors)

# ---------------------------------------------------------------------------
# 5. AUC (rank-based)
# ---------------------------------------------------------------------------
def auc(y_true, y_score):
    y_true = np.asarray(y_true)
    ranks = pd.Series(y_score).rank().values
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

auc_naive = auc(holdout["left_within_6mo"], naive_holdout_pred)
auc_full = auc(holdout["left_within_6mo"], full_holdout_pred)

# ---------------------------------------------------------------------------
# 6. Risk deciles (holdout, full model) + top-20% capture rate
# ---------------------------------------------------------------------------
risk = holdout.copy()
risk["pred"] = full_holdout_pred
risk["decile"] = pd.qcut(risk["pred"], 10, labels=False, duplicates="drop") + 1
decile_table = risk.groupby("decile", observed=True).agg(
    n=("left_within_6mo", "size"),
    actual_rate=("left_within_6mo", "mean"),
).reset_index()

n_deciles = risk["decile"].max()
top20_cutoff = n_deciles - 2  # top 2 deciles out of 10 (e.g., deciles 9 and 10)
top20 = risk[risk["decile"] > top20_cutoff]
capture_rate = top20["left_within_6mo"].sum() / risk["left_within_6mo"].sum()
top20_share_of_workforce = len(top20) / len(risk)

# ---------------------------------------------------------------------------
# 7. Write text output
# ---------------------------------------------------------------------------
lines = []
lines.append(f"N = {N}, train = {len(train)}, holdout = {len(holdout)}")
lines.append(f"Base 6-month turnover rate (full sample): {df['left_within_6mo'].mean():.1%}")
lines.append("")
lines.append("=== Naive model (engagement score only) ===")
lines.append(str(naive_model.summary2().tables[1]))
lines.append(f"Holdout AUC (naive): {auc_naive:.3f}")
lines.append("")
lines.append("=== Full model (+ comp ratio, promotion stall, peer attrition) ===")
lines.append(str(full_model.summary2().tables[1]))
lines.append(f"Holdout AUC (full): {auc_full:.3f}")
lines.append("")
lines.append("Odds ratios (full model):")
odds = np.exp(full_model.params)
ci = np.exp(full_model.conf_int())
odds_table = pd.DataFrame({"odds_ratio": odds, "ci_low": ci[0], "ci_high": ci[1]})
lines.append(str(odds_table))
lines.append("")
lines.append("Risk deciles (holdout, full model):")
lines.append(str(decile_table))
lines.append("")
lines.append(f"Top 2 deciles (~{top20_share_of_workforce:.0%} of holdout workforce) capture "
             f"{capture_rate:.0%} of actual 6-month leavers.")

output_text = "\n".join(lines)
print(output_text)

with open("analysis/turnover_risk_output.txt", "w") as f:
    f.write(output_text)

# ---------------------------------------------------------------------------
# 8. Charts — styled to match the site (generous margins from the start)
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
fig, ax = plt.subplots(figsize=(7, 4.6))
bars = ax.bar(["Engagement score\nalone", "Full model\n(+ comp, stalling, peer attrition)"],
               [auc_naive, auc_full], color=[MUTED, NAVY], width=0.55)
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color=BORDER, linewidth=1, linestyle="--")
ax.set_ylabel("Holdout AUC")
ax.set_title("Discrimination: engagement alone vs. full model", fontsize=13, weight="bold", pad=14)
for bar, val in zip(bars, [auc_naive, auc_full]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
            ha="center", fontsize=12, weight="bold", color="#111827")
fig.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.16)
fig.savefig(f"{IMG_DIR}/turnover_auc_comparison.svg")
plt.close(fig)

# Chart 2: Odds ratio forest plot (full model, excluding constant)
plot_odds = odds_table.drop(index="const").copy()
# Rescale to interpretable units
plot_odds.loc["comp_ratio"] = plot_odds.loc["comp_ratio"] ** 0.1        # per 0.1 comp-ratio unit
plot_odds.loc["months_since_promotion"] = plot_odds.loc["months_since_promotion"] ** 12  # per 12 months
plot_odds.loc["peer_attrition_rate"] = plot_odds.loc["peer_attrition_rate"] ** 0.1        # per 10 pts

label_map = {
    "engagement": "Engagement score (+1 pt)",
    "comp_ratio": "Pay vs. market (+0.1 ratio)",
    "months_since_promotion": "Time since promotion (+12 mo)",
    "peer_attrition_rate": "Team attrition, trailing 6mo (+10 pts)",
}
plot_odds = plot_odds.rename(index=label_map)
plot_odds = plot_odds.iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 4.8))
y_pos = np.arange(len(plot_odds))
ax.errorbar(plot_odds["odds_ratio"], y_pos,
            xerr=[plot_odds["odds_ratio"] - plot_odds["ci_low"],
                  plot_odds["ci_high"] - plot_odds["odds_ratio"]],
            fmt="o", color=NAVY, ecolor=MUTED, capsize=4, markersize=7)
ax.axvline(1.0, color=BORDER, linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_odds.index, fontsize=11)
ax.set_xlabel("Odds ratio (95% CI) — leaving within 6 months")
ax.set_title("What predicts turnover, and how strongly", fontsize=13, weight="bold", pad=14)
fig.subplots_adjust(left=0.34, right=0.96, top=0.87, bottom=0.14)
fig.savefig(f"{IMG_DIR}/turnover_odds_ratios.svg")
plt.close(fig)

# Chart 3: Risk deciles (holdout, full model)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(decile_table["decile"].astype(str), decile_table["actual_rate"], color=NAVY, width=0.65)
overall_rate = risk["left_within_6mo"].mean()
ax.axhline(overall_rate, color=BLUE, linestyle="--", linewidth=1.5)
ax.text(0.3, overall_rate + 0.012, f"Average rate ({overall_rate:.0%})", fontsize=10, color=BLUE)
ax.set_xlabel("Risk decile (1 = lowest predicted risk, 10 = highest)")
ax.set_ylabel("Actual 6-month turnover rate")
ax.set_title("Risk deciles: does the highest-risk group actually leave more?", fontsize=13, weight="bold", pad=14)
fig.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.14)
fig.savefig(f"{IMG_DIR}/turnover_risk_deciles.svg")
plt.close(fig)

print("\nCharts written to", IMG_DIR)
