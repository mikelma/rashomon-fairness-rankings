import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import Optional
from tyro.extras import SubcommandApp

app = SubcommandApp()


@app.command
def lp_results(dataset: str = "german.csv", sens_attr: str = "sex"):
    df = pd.read_csv(dataset)

    df = df[df["sens attr"] == sens_attr]
    # df = df[(df["eps"] < 100)]

    target, bayes, new = (
        df["target group ratio"],
        df["bayes group ratio"],
        df["new group ratio"],
    )
    df["improvement"] = 100 * ((target - bayes).abs() - (target - new).abs()) / target

    sns.lineplot(data=df, x="k", y="improvement", hue="eps")
    # sns.lineplot(data=df, x="k", y="new group ratio", hue="eps")
    # sns.lineplot(data=df, x="k", y="bayes group ratio", hue="eps")
    plt.xlabel("K")
    plt.ylabel("Fairness improvement (%)")
    plt.show()


@app.command
def score_distrib(
    dataset: str = "data/german_categorical-binsensitive.csv",
    score_attr: str = "month",
    sens_attr: str = "sex",
    n: Optional[int] = None,
):
    df = pd.read_csv(dataset)

    if n is None:
        n = len(df)
    print(n)

    lst_scores = df[score_attr][:n].values
    lst_scores = (lst_scores - lst_scores.min()) / (lst_scores.max() - lst_scores.min())
    lst_scores = 1 - lst_scores

    ranking = np.argsort(-lst_scores)

    cols = np.array(["tab:blue", "tab:orange"])
    groups = df[sens_attr].values
    colors = cols[(groups == 1).astype(int)]

    plt.bar(np.arange(n), lst_scores[ranking], color=colors[ranking])
    plt.ylabel("Score")
    custom_lines = [Line2D([0], [0], color=c) for c in cols]
    plt.legend(custom_lines, range(len(colors)), title=sens_attr)
    plt.show()

    ratios = []
    ordered_groups = groups[ranking]
    ks = np.arange(1, n)
    for k in ks:
        ratios.append(ordered_groups[:k].mean())

    tgt_ratio = groups.mean()

    plt.plot(100 * ks / n, ratios)
    plt.axhline(y=tgt_ratio, linestyle="dashed", color="tab:gray", label="True ratio")
    plt.xlabel("K (%)")
    plt.ylabel("Group 1 ratio")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    app.cli()
