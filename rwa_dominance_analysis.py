"""
Relative Weights Analysis (RWA) and Dominance Analysis
for Exit Survey Data (rwa_exit.csv)

This script performs both analyses on two outcome variables:
  1. recommendCompany
  2. returnToCompany

Predictors: supvExpectations, supvFeedback, supvRecognition, supvCared
"""

import numpy as np
import pandas as pd
from itertools import combinations
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ==============================================================================
# Load Data
# ==============================================================================
df = pd.read_csv("/home/user/hrmeasured/rwa_exit.csv")
predictors = ["supvExpectations", "supvFeedback", "supvRecognition", "supvCared"]
outcomes = ["returnToCompany", "recommendCompany"]

print("=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"N = {len(df)}")
print()
print(df[outcomes + predictors].describe().round(3))
print()

# Correlation matrix
print("Correlation Matrix:")
print(df[outcomes + predictors].corr().round(3))
print()


# ==============================================================================
# RELATIVE WEIGHTS ANALYSIS (RWA)
# ==============================================================================
# Implementation follows Johnson (2000) / Kabacoff (2015) methodology:
# 1. Compute correlation matrix of predictors (Rxx) and predictor-criterion
#    correlations (Rxy)
# 2. Eigendecompose Rxx to create orthogonal predictors (Lambda)
# 3. Regress criterion on orthogonal predictors to get beta weights
# 4. Transform back to get raw relative weights
# 5. Express as % of R-squared

def relative_weights_analysis(df, outcome, predictors):
    """
    Perform Relative Weights Analysis (Johnson, 2000).
    Returns a DataFrame with raw weights and % of R-squared.
    """
    # Full correlation matrix: outcome first, then predictors
    vars_all = [outcome] + predictors
    R = df[vars_all].corr().values

    nvar = len(vars_all)
    rxx = R[1:nvar, 1:nvar]  # predictor intercorrelations
    rxy = R[1:nvar, 0]       # predictor-criterion correlations

    # Eigenvalue decomposition of predictor correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(rxx)
    # Sort descending (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Create Lambda matrix
    delta = np.diag(np.sqrt(eigenvalues))
    lam = eigenvectors @ delta @ eigenvectors.T
    lambdasq = lam ** 2

    # Regression of criterion on orthogonal predictors
    beta = np.linalg.solve(lam, rxy)
    rsquare = np.sum(beta ** 2)

    # Raw relative weights
    rawwgt = lambdasq @ (beta ** 2)

    # Rescaled weights (% of R-squared)
    pct_rsquare = (rawwgt / rsquare) * 100

    results = pd.DataFrame({
        "Predictor": predictors,
        "Raw_Weight": rawwgt,
        "Pct_of_Rsquare": pct_rsquare
    }).sort_values("Pct_of_Rsquare", ascending=False).reset_index(drop=True)

    return results, rsquare


print("=" * 70)
print("RELATIVE WEIGHTS ANALYSIS (RWA)")
print("=" * 70)

rwa_results = {}
for outcome in outcomes:
    results, rsq = relative_weights_analysis(df, outcome, predictors)
    rwa_results[outcome] = (results, rsq)

    label = "Return to Company" if outcome == "returnToCompany" else "Recommend Company"
    print(f"\n--- {label} ---")
    print(f"Total R-squared: {rsq:.4f}")
    print()
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    # Verify weights sum to 100%
    print(f"Sum of % weights: {results['Pct_of_Rsquare'].sum():.2f}%")
    print()


# ==============================================================================
# DOMINANCE ANALYSIS
# ==============================================================================
# Dominance analysis (Budescu, 1993; Azen & Budescu, 2003) determines the
# relative importance of predictors by examining the change in R-squared when
# each predictor is added to ALL possible subset models.
#
# Three levels of dominance:
# 1. Complete dominance: X1 dominates X2 if adding X1 to any subset model
#    always yields a larger R-sq increase than adding X2
# 2. Conditional dominance: X1 dominates X2 if the average R-sq contribution
#    of X1 is larger than X2 at every model size
# 3. General dominance: X1 dominates X2 if the overall average R-sq
#    contribution of X1 is larger than X2

def get_rsquared(df, outcome, pred_subset):
    """Fit OLS and return R-squared for a given predictor subset."""
    if len(pred_subset) == 0:
        return 0.0
    X = sm.add_constant(df[list(pred_subset)])
    y = df[outcome]
    model = sm.OLS(y, X).fit()
    return model.rsquared


def dominance_analysis(df, outcome, predictors):
    """
    Perform dominance analysis (Budescu, 1993; Azen & Budescu, 2003).

    Returns:
      - general_dominance: DataFrame with general dominance statistics
      - conditional_dominance: DataFrame with conditional dominance (avg contribution by model size)
      - complete_dominance: dict of pairwise complete dominance results
      - all_subsets: dict of all subset R-squared values
    """
    p = len(predictors)

    # Step 1: Compute R-squared for all possible predictor subsets
    all_subsets = {}
    for size in range(0, p + 1):
        for combo in combinations(predictors, size):
            key = frozenset(combo)
            all_subsets[key] = get_rsquared(df, outcome, combo)

    # Step 2: Compute incremental R-squared for each predictor at each model size
    # For each predictor, compute the avg R-sq gain from adding it to subsets of size k
    incremental = {pred: {k: [] for k in range(p)} for pred in predictors}

    for pred in predictors:
        other_preds = [x for x in predictors if x != pred]
        for size in range(0, p):
            # All subsets of 'other_preds' of this size
            for combo in combinations(other_preds, size):
                subset_without = frozenset(combo)
                subset_with = frozenset(combo) | {pred}
                r2_without = all_subsets[subset_without]
                r2_with = all_subsets[subset_with]
                delta_r2 = r2_with - r2_without
                incremental[pred][size].append(delta_r2)

    # Step 3: Conditional dominance (average at each model size)
    conditional = {}
    for pred in predictors:
        conditional[pred] = {}
        for size in range(p):
            conditional[pred][size] = np.mean(incremental[pred][size])

    conditional_df = pd.DataFrame(conditional).T
    conditional_df.columns = [f"k={k}" for k in range(p)]

    # Step 4: General dominance (average of conditional dominance values)
    # Per Azen & Budescu (2003): D_i = (1/p) * sum of C_i(k) for k=0..p-1
    # Each model size gets equal weight
    general = {}
    for pred in predictors:
        general[pred] = np.mean([conditional[pred][k] for k in range(p)])

    general_df = pd.DataFrame({
        "Predictor": predictors,
        "General_Dominance": [general[p_] for p_ in predictors]
    })

    # Compute percentage of model R-squared
    full_r2 = all_subsets[frozenset(predictors)]
    general_df["Pct_of_Rsquare"] = (general_df["General_Dominance"] / full_r2) * 100
    general_df = general_df.sort_values("General_Dominance", ascending=False).reset_index(drop=True)

    # Step 5: Complete dominance (pairwise)
    complete_dom = {}
    for i, p1 in enumerate(predictors):
        for j, p2 in enumerate(predictors):
            if i >= j:
                continue
            p1_always_greater = True
            p2_always_greater = True
            for size in range(p):
                other_preds = [x for x in predictors if x != p1 and x != p2]
                for combo in combinations(other_preds, min(size, len(other_preds))):
                    if len(combo) != size:
                        continue
                    base = frozenset(combo)
                    r2_add_p1 = all_subsets[base | {p1}] - all_subsets[base]
                    r2_add_p2 = all_subsets[base | {p2}] - all_subsets[base]
                    if r2_add_p1 <= r2_add_p2:
                        p1_always_greater = False
                    if r2_add_p2 <= r2_add_p1:
                        p2_always_greater = False

            if p1_always_greater:
                complete_dom[(p1, p2)] = f"{p1} dominates {p2}"
            elif p2_always_greater:
                complete_dom[(p1, p2)] = f"{p2} dominates {p1}"
            else:
                complete_dom[(p1, p2)] = "No complete dominance"

    return general_df, conditional_df, complete_dom, full_r2


print("\n" + "=" * 70)
print("DOMINANCE ANALYSIS")
print("=" * 70)

dom_results = {}
for outcome in outcomes:
    general_df, conditional_df, complete_dom, full_r2 = dominance_analysis(df, outcome, predictors)
    dom_results[outcome] = (general_df, conditional_df, complete_dom, full_r2)

    label = "Return to Company" if outcome == "returnToCompany" else "Recommend Company"
    print(f"\n--- {label} ---")
    print(f"Full Model R-squared: {full_r2:.4f}")

    print(f"\nGeneral Dominance Statistics:")
    print(general_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSum of % weights: {general_df['Pct_of_Rsquare'].sum():.2f}%")

    print(f"\nConditional Dominance (average delta R-sq by model size k):")
    print(conditional_df.round(4).to_string())

    print(f"\nComplete Dominance (pairwise):")
    for pair, result in complete_dom.items():
        print(f"  {pair[0]} vs {pair[1]}: {result}")
    print()


# ==============================================================================
# COMPARISON: RWA vs DOMINANCE ANALYSIS
# ==============================================================================
print("\n" + "=" * 70)
print("COMPARISON: RWA vs DOMINANCE ANALYSIS")
print("=" * 70)

for outcome in outcomes:
    label = "Return to Company" if outcome == "returnToCompany" else "Recommend Company"
    print(f"\n--- {label} ---")

    rwa_res, rwa_rsq = rwa_results[outcome]
    dom_gen, dom_cond, dom_comp, dom_rsq = dom_results[outcome]

    # Build comparison table
    comp = rwa_res[["Predictor", "Raw_Weight", "Pct_of_Rsquare"]].copy()
    comp.columns = ["Predictor", "RWA_Raw_Weight", "RWA_Pct"]

    dom_map = dict(zip(dom_gen["Predictor"], dom_gen["General_Dominance"]))
    dom_pct_map = dict(zip(dom_gen["Predictor"], dom_gen["Pct_of_Rsquare"]))

    comp["DA_General_Dom"] = comp["Predictor"].map(dom_map)
    comp["DA_Pct"] = comp["Predictor"].map(dom_pct_map)
    comp["Diff_Pct"] = comp["RWA_Pct"] - comp["DA_Pct"]

    # Rank
    comp["RWA_Rank"] = comp["RWA_Pct"].rank(ascending=False).astype(int)
    comp["DA_Rank"] = comp["DA_Pct"].rank(ascending=False).astype(int)
    comp["Rank_Match"] = comp["RWA_Rank"] == comp["DA_Rank"]

    print(f"\nR-squared: RWA={rwa_rsq:.4f}, DA={dom_rsq:.4f}")
    print()
    print(comp[["Predictor", "RWA_Pct", "DA_Pct", "Diff_Pct", "RWA_Rank", "DA_Rank", "Rank_Match"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"
    ))
    print()

    # Check rank agreement
    if all(comp["Rank_Match"]):
        print("  -> Rank ordering is IDENTICAL between RWA and DA")
    else:
        print("  -> Rank ordering DIFFERS between RWA and DA")
        mismatches = comp[~comp["Rank_Match"]]
        for _, row in mismatches.iterrows():
            print(f"     {row['Predictor']}: RWA rank={row['RWA_Rank']}, DA rank={row['DA_Rank']}")

    # Max absolute difference in % weights
    max_diff = comp["Diff_Pct"].abs().max()
    print(f"  -> Maximum absolute difference in % weights: {max_diff:.2f}%")
    print()


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
# Inspired by AJ Thurston's dominance analysis visualization approach
# (https://github.com/AJThurston/dominance)

# Readable predictor labels
pred_labels = {
    "supvExpectations": "Set Clear\nExpectations",
    "supvFeedback": "Provided Timely\nFeedback",
    "supvRecognition": "Recognized\nGood Work",
    "supvCared": "Cared as\na Person",
}
pred_labels_oneline = {
    "supvExpectations": "Set Clear Expectations",
    "supvFeedback": "Provided Timely Feedback",
    "supvRecognition": "Recognized Good Work",
    "supvCared": "Cared as a Person",
}
outcome_labels = {
    "returnToCompany": "Return to Company",
    "recommendCompany": "Recommend Company",
}

colors = ["#336666", "#ae98d7", "#262626", "#7A918D"]

# --------------------------------------------------------------------------
# FIGURE 1: General Dominance - Thurston-style Stacked Horizontal Bar
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.5))

bar_height = 0.35
y_positions = [1, 0]  # Return on top, Recommend on bottom

for idx, outcome in enumerate(outcomes):
    gen_df = dom_results[outcome][0].copy()
    full_r2 = dom_results[outcome][3]
    # Sort by general dominance descending for consistent stacking
    gen_df = gen_df.sort_values("General_Dominance", ascending=False).reset_index(drop=True)

    left = 0
    for j, row in gen_df.iterrows():
        width = row["General_Dominance"]
        bar = ax.barh(y_positions[idx], width, height=bar_height, left=left,
                      color=colors[j], edgecolor="white", linewidth=0.5)
        # Label inside the bar segment
        cx = left + width / 2
        pct = row["Pct_of_Rsquare"]
        label = f"{pred_labels_oneline[row['Predictor']]}\n{pct:.1f}%"
        ax.text(cx, y_positions[idx], label, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
        left += width

    # Total R² label at the end
    ax.text(left + 0.008, y_positions[idx],
            f"Total: {full_r2:.1%}",
            ha="left", va="center", fontsize=9, fontweight="bold", color="#333333")

ax.set_yticks(y_positions)
ax.set_yticklabels([outcome_labels[o] for o in outcomes], fontsize=10, fontweight="bold")
ax.set_xlim(0, 0.50)
ax.set_xlabel(r"Percentage of $R^2$ Accounted for by Predictor", fontsize=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("General Dominance Weights: Supervisor Behaviors Predicting Exit Survey Outcomes",
             fontsize=11, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout()
plt.savefig("/home/user/hrmeasured/da_general_dominance_stacked.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: da_general_dominance_stacked.png")

# --------------------------------------------------------------------------
# FIGURE 2: General Dominance - Grouped Horizontal Bar Chart
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

pred_order = dom_results["recommendCompany"][0].sort_values(
    "General_Dominance", ascending=True)["Predictor"].tolist()

y = np.arange(len(pred_order))
bar_h = 0.35

for i, outcome in enumerate(outcomes):
    gen_df = dom_results[outcome][0]
    vals = [gen_df.loc[gen_df["Predictor"] == p, "Pct_of_Rsquare"].values[0]
            for p in pred_order]
    offset = -bar_h / 2 if i == 0 else bar_h / 2
    bars = ax.barh(y + offset, vals, height=bar_h, color=colors[i],
                   label=outcome_labels[outcome], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8.5, color="#333333")

ax.set_yticks(y)
ax.set_yticklabels([pred_labels_oneline[p] for p in pred_order], fontsize=9)
ax.set_xlabel(r"% of $R^2$", fontsize=10)
ax.set_title("General Dominance: Predictor Importance by Outcome",
             fontsize=11, fontweight="bold", pad=10)
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 35)
plt.tight_layout()
plt.savefig("/home/user/hrmeasured/da_general_dominance_grouped.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: da_general_dominance_grouped.png")

# --------------------------------------------------------------------------
# FIGURE 3: Conditional Dominance - Line Plot
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax_idx, outcome in enumerate(outcomes):
    ax = axes[ax_idx]
    cond_df = dom_results[outcome][1]

    for j, pred in enumerate(predictors):
        vals = cond_df.loc[pred].values
        model_sizes = range(len(vals))
        ax.plot(model_sizes, vals, marker="o", color=colors[j], linewidth=2,
                markersize=7, label=pred_labels_oneline[pred])

    ax.set_xlabel("Number of Predictors Already in Model (k)", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel(r"Average Incremental $R^2$", fontsize=10)
    ax.set_title(outcome_labels[outcome], fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(predictors)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", frameon=True)

fig.suptitle("Conditional Dominance: Average Incremental Contribution by Model Size",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/home/user/hrmeasured/da_conditional_dominance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: da_conditional_dominance.png")

# --------------------------------------------------------------------------
# FIGURE 4: Complete Dominance - Pairwise Heatmap
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, outcome in enumerate(outcomes):
    ax = axes[ax_idx]
    comp_dom = dom_results[outcome][2]
    gen_df = dom_results[outcome][0]
    # Order predictors by general dominance (descending)
    ordered = gen_df.sort_values("General_Dominance", ascending=False)["Predictor"].tolist()
    n = len(ordered)

    # Build matrix: 1 = row dominates col, -1 = col dominates row, 0 = no complete dominance
    mat = np.full((n, n), np.nan)
    for i in range(n):
        mat[i, i] = 0  # diagonal
        for j in range(n):
            if i == j:
                continue
            # Find the pair in complete_dom
            key1 = (ordered[i], ordered[j])
            key2 = (ordered[j], ordered[i])
            result = comp_dom.get(key1, comp_dom.get(key2, None))
            if result is None:
                mat[i, j] = 0
            elif f"{ordered[i]} dominates" in result:
                mat[i, j] = 1
            elif f"{ordered[j]} dominates" in result:
                mat[i, j] = -1
            else:
                mat[i, j] = 0

    cmap = ListedColormap(["#d9534f", "#f0f0f0", "#5cb85c"])
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    short_labels = [pred_labels_oneline[p] for p in ordered]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=7.5)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="#999")
            elif mat[i, j] == 1:
                ax.text(j, i, "Dom", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            elif mat[i, j] == -1:
                ax.text(j, i, "Sub", ha="center", va="center", fontsize=8,
                        color="#333", fontweight="bold")
            else:
                ax.text(j, i, "No", ha="center", va="center", fontsize=8, color="#666")

    ax.set_title(outcome_labels[outcome], fontsize=11, fontweight="bold")
    ax.set_xlabel("Column Predictor", fontsize=9)
    if ax_idx == 0:
        ax.set_ylabel("Row Predictor", fontsize=9)

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#5cb85c", label="Row dominates column"),
    mpatches.Patch(facecolor="#f0f0f0", edgecolor="#ccc", label="No complete dominance"),
    mpatches.Patch(facecolor="#d9534f", label="Column dominates row"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("Complete Dominance: Pairwise Comparisons",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/home/user/hrmeasured/da_complete_dominance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: da_complete_dominance.png")


# ==============================================================================
# METHODOLOGICAL COMPARISON
# ==============================================================================
print("\n" + "=" * 70)
print("METHODOLOGICAL COMPARISON: RWA vs DOMINANCE ANALYSIS")
print("=" * 70)
print("""
RELATIVE WEIGHTS ANALYSIS (RWA) - Johnson (2000)
-------------------------------------------------
Approach: Creates a set of orthogonal (uncorrelated) predictors that are
maximally related to the original predictors via eigenvalue decomposition
of the predictor correlation matrix. Regression weights from these
orthogonal predictors are then transformed back to estimate each original
predictor's proportionate contribution to R-squared.

Strengths:
  - Computationally efficient (no subset models needed)
  - Produces a single, clean decomposition of R-squared
  - Weights always sum exactly to model R-squared
  - Well-suited for large numbers of predictors

Limitations:
  - Relies on a specific mathematical transformation; less transparent
  - Does not provide pairwise dominance comparisons
  - Approximation of true relative importance

DOMINANCE ANALYSIS (DA) - Budescu (1993)
-----------------------------------------
Approach: Examines the incremental contribution of each predictor across
ALL possible subset models (2^p subsets for p predictors). Three levels
of dominance are assessed:
  - Complete: X1 always contributes more than X2 in every subset
  - Conditional: X1's average contribution is larger at every model size
  - General: X1's overall average contribution is larger

Strengths:
  - Most thorough: examines all possible model configurations
  - Provides three levels of dominance (complete, conditional, general)
  - More transparent and interpretable
  - Considered the "gold standard" for relative importance

Limitations:
  - Computationally expensive: 2^p models required (impractical for p > ~15)
  - General dominance weights may differ slightly from RWA weights
  - Complete dominance may not exist for all predictor pairs

WHICH IS BETTER?
-----------------
For this specific analysis with 4 predictors (16 total subset models):

Dominance analysis is the stronger choice because:
  1. With only 4 predictors, the computational cost is trivial (16 models)
  2. DA provides richer information: complete, conditional, AND general
     dominance - not just a single weight
  3. Complete and conditional dominance provide stronger claims about
     predictor importance than RWA's single percentage
  4. DA is considered the gold standard in the relative importance
     literature (Azen & Budescu, 2003; Braun & Oswald, 2011)
  5. The general dominance weights decompose R-squared exactly, just
     like RWA

That said, both methods produce very similar rank orderings and
percentage weights, which is expected when predictors have moderate
intercorrelations. The convergence of results across methods strengthens
confidence in the findings.

For practical use:
  - Use DA when p is small (< ~15) - you get more information
  - Use RWA when p is large - it scales much better
  - When both are feasible, report both for robustness

References:
  - Budescu, D. V. (1993). Dominance analysis. Psychological Bulletin, 114, 542-551.
  - Johnson, J. W. (2000). A heuristic method for estimating the relative
    weight of predictor variables in multiple regression. Multivariate
    Behavioral Research, 35, 1-19.
  - Azen, R., & Budescu, D. V. (2003). The dominance analysis approach for
    comparing predictors in multiple regression. Psychological Methods, 8, 129-148.
  - Braun, M. T., & Oswald, F. L. (2011). Exploratory regression analysis.
    Behavior Research Methods, 43, 331-339.
""")
