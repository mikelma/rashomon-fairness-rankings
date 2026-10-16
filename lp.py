import pulp as pl
import pandas as pd
import numpy as np

n = 20
k = 10
eps = 10000

df = pd.read_csv("data/german_categorical-binsensitive.csv")

lst_groups = df["age"][:n]
num_groups = len(lst_groups.unique())
groups = range(num_groups)  # set of sensitive groups
gs = lst_groups.value_counts()
group_sizes = [int(gs[g]) for g in groups]
lst_groups = lst_groups.values

individuals = range(n)  # set of individuals (IDs)

# normalize scores
lst_scores = df["month"][:n].values
lst_scores = (lst_scores - lst_scores.min()) / (lst_scores.max() - lst_scores.min())
lst_scores = 1 - lst_scores

print("scores:", lst_scores)

ids = np.arange(n)
ranking = np.argsort(-lst_scores)
opt_top_k = ranking[:k]


prob = pl.LpProblem("FairTopK", pl.LpMinimize)

#
# ---- Problem definition ---- #
#
alpha = prob.add_variable_dict("alphas", individuals, 0, 1, pl.LpBinary)

delta_plus = prob.add_variable_dict("delta_plus", groups, 0, None, pl.LpContinuous)
delta_minus = prob.add_variable_dict("delta_minus", groups, 0, None, pl.LpContinuous)

#
# ---- Constraints ---- #
#
# >>> Constr. (1)
prob += pl.lpSum(alpha) == k

# >>> Constr. (2)
in_top_k_mask = np.isin(ids, opt_top_k)
# lists of IDs that are/aren't contained in the optimal top-K
in_top_k = ids[in_top_k_mask]
out_top_k = ids[~in_top_k_mask]
prob += (
    pl.lpSum([lst_scores[i] * (1 - alpha[i]) for i in in_top_k])
    - pl.lpSum([lst_scores[j] * alpha[j] for j in out_top_k])
) <= eps
prob += (
    pl.lpSum([lst_scores[i] * (1 - alpha[i]) for i in in_top_k])
    - pl.lpSum([lst_scores[j] * alpha[j] for j in out_top_k])
) >= 0

# >>> Constr. (3)
# this is the variable `a` in the paper
# in this case `a_{g,i}` instead of `a_{i,g}`
group_belong = [np.asarray(lst_groups == g).astype(float) for g in groups]
for g in groups:
    g_size = sum(group_belong[g])
    delta_deltas = delta_plus[g] - delta_minus[g]
    s = pl.lpSum(group_belong[g][i] * alpha[i] for i in individuals)
    prob += (1 / k) * s - (g_size / n) == delta_deltas

# >>> Objective
# prob += pl.lpSum(
#     list(delta_plus.values()) + list(delta_minus.values())
# )  # sum everything
prob += pl.lpSum([delta_plus[g] + delta_minus[g] for g in groups])  # sum everything

stats = prob.solve(pl.GLPK())


prob.writeLP("debug_model.lp")

#
# ---- Process the solution ---- #
#
print(stats)

# print()
# print([pl.value(v) for v in delta_plus.values()])
# print([pl.value(dg) for dg in delta_plus])
# print([pl.value(dg) for dg in delta_minus])
# quit()

alpha_values = [int(var.varValue) for var in alpha.values()]
delta_plus_values = [var.varValue for var in delta_plus.values()]
delta_minus_values = [var.varValue for var in delta_minus.values()]

# Print them as lists
print(f"\nAlpha values:      {alpha_values}")
print(f"Delta plus values: {delta_plus_values}")
print(f"Delta minus values:{delta_minus_values}")

selected_ids = [i for i, val in enumerate(alpha_values) if val == 1]
print(f"\nIndividual IDs for original (optimal) Top-K: {opt_top_k.tolist()}")
print(f"Selected Individual IDs for Top-K: {selected_ids}")

print()
print(f"Bayes group ratio: {lst_groups[opt_top_k].mean()}")
print(f"Optimized group ratio: {lst_groups[np.array(selected_ids)].mean()}")
print(f"Target ratio (original): {np.array(group_sizes) / n}")

print()
print(
    "constraint 2:",
    sum([lst_scores[i] * (1 - alpha_values[i]) for i in in_top_k])
    - sum([lst_scores[j] * alpha_values[j] for j in out_top_k]),
)
print()
print(
    "constraint 3:",
    [
        (1 / k) * np.dot(group_belong[g], np.array(alpha_values)) - (group_sizes[g] / n)
        for g in groups
    ],
)

score_bayes_top_k = lst_scores[opt_top_k].sum()
score_new_top_k = lst_scores[np.array(selected_ids)].sum()
print("Utility bayes top-K:", score_bayes_top_k)
print("Utility new top-K:  ", score_new_top_k)
