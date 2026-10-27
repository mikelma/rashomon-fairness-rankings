import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
from typing import Optional
from tyro.extras import SubcommandApp

app = SubcommandApp()


@app.command
def lp_results(
    dataset: str = "german.csv", sens_attr: str = "sex", font_size: int = 16
):
    matplotlib.rcParams.update(
        {
            "font.size": font_size,
            "font.family": "serif",
            "text.usetex": True,
            "pdf.fonttype": 42,
        }
    )

    df = pd.read_csv(dataset)

    df = df[df["sens attr"] == sens_attr]

    target, bayes, new = (
        df["target group ratio"],
        df["bayes group ratio"],
        df["new group ratio"],
    )
    df["improvement"] = 100 * ((target - bayes).abs() - (target - new).abs()) / target

    print(
        df[df["improvement"] < 0][
            [
                "k",
                "n",
                "eps",
                "bayes group ratio",
                "new group ratio",
                "target group ratio",
                "improvement",
            ]
        ]
    )

    # df = df[(df["eps"] == 1)]
    # df = df[(df["k"] > 10)]

    sns.lineplot(
        data=df,
        x="k",
        y="improvement",
        hue="eps",
        palette=sns.color_palette("tab10"),
        linewidth=3,
        alpha=0.7,
        style="eps",
        markers=True,
        markersize=10,
    )

    # for eps in df["eps"].unique():
    #     sel = df[df["eps"] == eps]
    #     g = plt.plot(
    #         sel["k"], sel["new group ratio"], label=eps, linewidth=3, alpha=0.6
    #     )
    #     c = g[0].get_color()
    #     plt.plot(sel["k"], sel["bayes group ratio"], color=c, linestyle="dashed")
    #     plt.plot(sel["k"], sel["target group ratio"], color=c, linestyle="dotted")

    plt.legend()

    plt.xlabel("K")
    plt.ylabel("Fairness improvement (\\%)")
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.show()

    # target_ratio = df["target group ratio"].values[0]

    # for eps in df["eps"].unique():
    #     sel = df[df["eps"] == eps]
    #     g = plt.plot(
    #         sel["k"], sel["new group ratio"], label=eps, linewidth=3, alpha=0.6
    #     )
    #     c = g[0].get_color()
    #     plt.plot(sel["k"], sel["bayes group ratio"], color=c, linestyle="dashed")
    # plt.axhline(y=target_ratio, color="tab:gray", linewidth=2, linestyle="dotted")
    # plt.show()


@app.command
def score_distrib(
    dataset: str = "data/german_categorical-binsensitive.csv",
    score_attr: str = "month",
    sens_attr: str = "sex",
    n: Optional[int] = None,
    font_size: int = 16,
):
    matplotlib.rcParams.update(
        {
            "font.size": font_size,
            "font.family": "serif",
            "text.usetex": True,
            "pdf.fonttype": 42,
        }
    )

    df = pd.read_csv(dataset)

    if n is None:
        n = len(df)
    print(n)

    lst_scores = df[score_attr][:n].values
    lst_scores = (lst_scores - lst_scores.min()) / (lst_scores.max() - lst_scores.min())
    lst_scores = 1 - lst_scores

    ranking = np.argsort(-lst_scores)

    # cols = np.array(["tab:blue", "tab:orange"])
    groups = df[sens_attr][:n].values
    # colors = cols[(groups == 1).astype(int)]

    # plt.bar(np.arange(n), lst_scores[ranking], color=colors[ranking])
    # custom_lines = [Line2D([0], [0], color=c) for c in cols]
    # plt.legend(custom_lines, range(len(colors)), title=sens_attr)
    plt.plot(np.arange(n), lst_scores[ranking])
    plt.ylabel("Score")
    plt.show()

    ratios = []
    ordered_groups = groups[ranking]
    ks = np.arange(int(n * 0.1), n)
    for k in ks:
        ratios.append(ordered_groups[:k].mean())

    tgt_ratio = groups.mean()

    plt.plot(100 * ks / n, ratios)
    plt.axhline(y=tgt_ratio, linestyle="dashed", color="tab:gray", label="True ratio")
    plt.xlabel("K (\\%)")
    plt.ylabel("Group 1 ratio")
    plt.legend()
    plt.show()

    grouped_scores = []
    for group in np.unique(groups):
        grouped_scores.append(lst_scores[groups == group])
    dfs = []
    for name, scores in [
        # ("all", lst_scores),
        ("0", grouped_scores[0]),
        ("1", grouped_scores[1]),
    ]:
        df = pd.DataFrame()
        df["score"] = scores
        df["group"] = name
        dfs.append(df)

    df = pd.concat(dfs)
    g = sns.kdeplot(data=df, x="score", hue="group")

    print(g)
    plt.show()


if __name__ == "__main__":
    app.cli()
