import pulp as pl
import pandas as pd
import numpy as np

n = 20
k = 5
eps = 15.0

df = pd.read_csv("data/german_categorical-binsensitive.csv")

lst_groups = df["sex"][:n]
num_groups = len(lst_groups.unique())
groups = range(num_groups)  # set of sensitive groups
gs = lst_groups.value_counts()
group_sizes = [int(gs[g]) for g in groups]
lst_groups = lst_groups.values

individuals = range(n)  # set of individuals (IDs)

lst_scores = df["month"][:n].values

ids = np.arange(n)
ranking = np.argsort(lst_scores)
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
in_top_k_mask = np.isin(ids, opt_top_k)
# lists of IDs that are/aren't contained in the optimal top-K
in_top_k = ids[in_top_k_mask]
out_top_k = ids[~in_top_k_mask]

# >>> Constr. (2)
prob += (
    pl.lpSum([lst_scores[i] * (1 - alpha[i]) for i in in_top_k])
    - pl.lpSum([lst_scores[j] * alpha[j] for j in out_top_k])
    <= eps
)

# >>> Constr. (3)
# this is the variable `a` in the paper
# in this case `a_{g,i}` instead of `a_{i,g}`
group_belong = [np.asarray(lst_groups == g).astype(float) for g in groups]
for g in groups:
    g_size = sum(group_belong[g])
    delta_deltas = delta_plus[g] - delta_minus[g]
    prob += ((1 / k) * pl.lpDot(group_belong[g], alpha)) - (g_size / n) == delta_deltas

# >>> Objective
prob += pl.lpSum(
    list(delta_plus.values()) + list(delta_minus.values())
)  # sum everything

stats = prob.solve(pl.GLPK())

print(stats)
print()
print(stats.status_str)
