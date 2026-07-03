"""
"High Potential" as a Measured Construct — synthetic demo analysis for the
HR Measured case study on measurement design (not prediction).

Illustrative, NOT a real client engagement. Every number in the case study
copy comes from the actual output of this script (printed below and saved to
analysis/high_potential_output.txt).

Design of the simulation:
- One true latent trait ("true potential") underlies three distinct,
  correlated-but-separable facets: learning agility, drive for scope, and
  interpersonal influence (facet structure loosely following Lombardo &
  Eichinger's learning-agility framing of high-potential identification).
- Two independent raters (manager + a calibration-panel rater) each rate all
  three facets on a 1-5 behaviorally-anchored scale. Facets differ in how
  reliably they can be observed — interpersonal behavior is easier for two
  observers to agree on than an inferential judgment like "drive."
- Reliability is assessed with the intraclass correlation coefficient (ICC),
  the standard tool for rater agreement (Shrout & Fleiss, 1979), interpreted
  against conventional benchmarks (Koo & Li, 2016).
- A naive, unstructured "gut feel" HiPo nomination is simulated separately:
  a single rating from a single rater, contaminated by a performance-halo
  effect unrelated to actual potential — modeling the real failure mode of
  unstructured nomination processes.
- Validity is checked against an independent 12-months-later criterion
  (a skip-level assessment not used to build the rubric), comparing the
  naive nomination to the structured composite.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260618)
N = 380  # employees in the talent-review population

# ---------------------------------------------------------------------------
# 1. True latent trait and three facets
# ---------------------------------------------------------------------------
true_potential = rng.normal(0, 1, N)

true_learning_agility = 0.75 * true_potential + rng.normal(0, 0.66, N)
true_drive = 0.65 * true_potential + rng.normal(0, 0.76, N)
true_interpersonal = 0.70 * true_potential + rng.normal(0, 0.71, N)

# ---------------------------------------------------------------------------
# 2. Two raters rate all three facets, each with facet-specific noise
#    (interpersonal behavior easiest to observe consistently; drive hardest)
# ---------------------------------------------------------------------------
def to_five_point(x, center=3.0, spread=0.9):
    return np.clip(np.round(center + x * spread), 1, 5)

rater_noise_sd = {
    "learning_agility": 0.55,
    "drive": 0.85,
    "interpersonal": 0.40,
}

facet_true = {
    "learning_agility": true_learning_agility,
    "drive": true_drive,
    "interpersonal": true_interpersonal,
}

ratings = {}
for facet, true_val in facet_true.items():
    sd = rater_noise_sd[facet]
    r1 = to_five_point(true_val + rng.normal(0, sd, N))
    r2 = to_five_point(true_val + rng.normal(0, sd, N))
    ratings[f"{facet}_rater1"] = r1
    ratings[f"{facet}_rater2"] = r2

df = pd.DataFrame(ratings)

# ---------------------------------------------------------------------------
# 3. Naive, unstructured single-rater "gut feel" HiPo nomination
#    Contaminated by a recent performance-halo effect unrelated to potential.
# ---------------------------------------------------------------------------
performance_halo = rng.normal(0, 1, N)
naive_component = 0.5 * true_potential + 0.45 * performance_halo
naive_rating = to_five_point(naive_component + rng.normal(0, 0.5, N))
df["naive_rating"] = naive_rating

# ---------------------------------------------------------------------------
# 4. Independent 12-months-later criterion (not used to build the rubric)
# ---------------------------------------------------------------------------
outcome = 0.60 * true_potential + rng.normal(0, 0.85, N)
df["outcome_12mo"] = outcome

# ---------------------------------------------------------------------------
# 5. ICC (Shrout & Fleiss, 1979), two-way random, absolute agreement
# ---------------------------------------------------------------------------
def icc_absolute(mat):
    """mat: n_subjects x k_raters array. Returns ICC(2,1) and ICC(2,k)."""
    n, k = mat.shape
    grand_mean = mat.mean()
    row_means = mat.mean(axis=1)
    col_means = mat.mean(axis=0)

    ssr = k * np.sum((row_means - grand_mean) ** 2)
    ssc = n * np.sum((col_means - grand_mean) ** 2)
    sst = np.sum((mat - grand_mean) ** 2)
    sse = sst - ssr - ssc

    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))

    icc_2_1 = (msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse))
    icc_2_k = (msr - mse) / (msr + (msc - mse) / n)
    return icc_2_1, icc_2_k

reliability = {}
for facet in facet_true:
    mat = df[[f"{facet}_rater1", f"{facet}_rater2"]].values
    icc1, icc2 = icc_absolute(mat)
    reliability[facet] = {"icc_single_rater": icc1, "icc_two_rater_avg": icc2}

# Composite: average of the three facets, per rater
df["composite_rater1"] = df[[f"{f}_rater1" for f in facet_true]].mean(axis=1)
df["composite_rater2"] = df[[f"{f}_rater2" for f in facet_true]].mean(axis=1)
comp_mat = df[["composite_rater1", "composite_rater2"]].values
comp_icc1, comp_icc2 = icc_absolute(comp_mat)
reliability["composite"] = {"icc_single_rater": comp_icc1, "icc_two_rater_avg": comp_icc2}

# Operational structured score: average across both raters and three facets
df["structured_score"] = df[["composite_rater1", "composite_rater2"]].mean(axis=1)

# ---------------------------------------------------------------------------
# 6. Validity: correlation with the independent 12-month criterion
# ---------------------------------------------------------------------------
def pearson_r_ci(x, y):
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return r, lo, hi

r_naive, lo_naive, hi_naive = pearson_r_ci(df["naive_rating"], df["outcome_12mo"])
r_structured, lo_structured, hi_structured = pearson_r_ci(df["structured_score"], df["outcome_12mo"])

# ---------------------------------------------------------------------------
# 7. Write text output
# ---------------------------------------------------------------------------
lines = []
lines.append(f"N = {N} employees, 2 raters, 3 facets")
lines.append("")
lines.append("=== Reliability (ICC, absolute agreement) by facet ===")
for facet, vals in reliability.items():
    lines.append(f"{facet}: ICC(single rater) = {vals['icc_single_rater']:.3f}, "
                 f"ICC(2-rater average) = {vals['icc_two_rater_avg']:.3f}")
lines.append("")
lines.append("=== Validity: correlation with independent 12-month criterion ===")
lines.append(f"Naive single-rater nomination: r = {r_naive:.3f} [{lo_naive:.3f}, {hi_naive:.3f}]")
lines.append(f"Structured composite (2 raters x 3 facets): r = {r_structured:.3f} "
             f"[{lo_structured:.3f}, {hi_structured:.3f}]")

output_text = "\n".join(lines)
print(output_text)

with open("analysis/high_potential_output.txt", "w") as f:
    f.write(output_text)

# ---------------------------------------------------------------------------
# 8. Charts — styled to match the site
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

# Chart 1: reliability by facet (single rater vs 2-rater average)
facet_labels = {"learning_agility": "Learning agility", "drive": "Drive for scope",
                 "interpersonal": "Interpersonal influence", "composite": "Composite (all 3 facets)"}
order = ["interpersonal", "learning_agility", "drive", "composite"]
single = [reliability[f]["icc_single_rater"] for f in order]
avg2 = [reliability[f]["icc_two_rater_avg"] for f in order]
labels = [facet_labels[f] for f in order]

fig, ax = plt.subplots(figsize=(8.5, 5.4))
y = np.arange(len(order))
h = 0.32
ax.barh(y + h / 2, single, height=h, color=MUTED, label="Single rater")
ax.barh(y - h / 2, avg2, height=h, color=NAVY, label="Average of 2 raters")
ax.axvline(0.75, color=BORDER, linestyle="--", linewidth=1)
ax.text(0.76, len(order) - 0.62, "\"good\" (Koo & Li, 2016)", fontsize=9.5, color=MUTED)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("ICC (absolute agreement)")
ax.set_title("Reliability by facet, and why averaging raters helps", fontsize=13, weight="bold", pad=38)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False, fontsize=10)
fig.subplots_adjust(left=0.30, right=0.96, top=0.80, bottom=0.12)
fig.savefig(f"{IMG_DIR}/hipo_reliability.svg")
plt.close(fig)

# Chart 2: naive vs structured validity comparison
fig, ax = plt.subplots(figsize=(6.5, 4.3))
bars = ax.bar(["Naive single-rater\nnomination", "Structured composite\n(2 raters x 3 facets)"],
               [r_naive, r_structured], color=[MUTED, NAVY], width=0.55)
ax.set_ylim(0, 0.7)
ax.set_ylabel("Correlation with 12-month outcome (r)")
ax.set_title("Validity: naive nomination vs. structured measurement", fontsize=13, weight="bold", pad=14)
for bar, val in zip(bars, [r_naive, r_structured]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"r = {val:.2f}",
            ha="center", fontsize=12, weight="bold", color="#111827")
fig.subplots_adjust(left=0.14, right=0.96, top=0.87, bottom=0.16)
fig.savefig(f"{IMG_DIR}/hipo_validity_comparison.svg")
plt.close(fig)

# Chart 3: validity scatter (structured score vs outcome) with regression line
fig, ax = plt.subplots(figsize=(6, 5.2))
ax.scatter(df["structured_score"], df["outcome_12mo"], alpha=0.45, s=22, color=NAVY, edgecolors="none")
slope, intercept = np.polyfit(df["structured_score"], df["outcome_12mo"], 1)
xs = np.linspace(df["structured_score"].min(), df["structured_score"].max(), 50)
ax.plot(xs, slope * xs + intercept, color=BLUE, linewidth=2)
ax.set_xlabel("Structured HiPo composite score (rubric)")
ax.set_ylabel("Independent 12-month outcome")
ax.set_title(f"Structured score vs. later outcome (r = {r_structured:.2f})", fontsize=13, weight="bold", pad=14)
fig.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.13)
fig.savefig(f"{IMG_DIR}/hipo_validity_scatter.svg")
plt.close(fig)

print("\nCharts written to", IMG_DIR)
