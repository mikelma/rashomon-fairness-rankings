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
    dataset: str = "german.csv",
    sens_attr: str = "sex",
    font_size: int = 16,
    save: bool = False,
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

    sns.lineplot(
        data=df,
        x="k",
        y="improvement",
        hue="eps",
        palette=sns.color_palette("tab10"),
        linewidth=3,
        # alpha=0.7,
        style="eps",
        markers=True,
        markersize=14,
    )
    # plt.legend()

    plt.xlabel("K")
    plt.ylabel("Fairness improvement (\\%)")
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.tight_layout()

    if save:
        name = dataset.split("/")[-1].split(".")[0]
        plt.savefig(f"figures/lp/{name}_{sens_attr}.pdf")

    plt.show()


@app.command
def score_distrib(
    dataset: str = "data/german_categorical-binsensitive.csv",
    score_attr: str = "month",
    sens_attr: str = "sex",
    n: Optional[int] = None,
    font_size: int = 16,
    save_dir: Optional[str] = None,
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

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    for ax in axs:
        ax.spines[["right", "top"]].set_visible(False)

    axs[0].plot(np.arange(n), lst_scores[ranking])
    axs[0].set_ylabel("Score")
    axs[0].set_xlabel("Individual (sorted)")

    ratios = []
    ordered_groups = groups[ranking]
    ks = np.arange(int(n * 0.1), n)
    for k in ks:
        ratios.append(ordered_groups[:k].mean())

    tgt_ratio = groups.mean()

    axs[1].plot(100 * ks / n, ratios)
    axs[1].axhline(
        y=tgt_ratio, linestyle="dashed", color="tab:gray", label="True ratio"
    )
    axs[1].set_xlabel("K (\\%)")
    axs[1].set_ylabel("Group 1 ratio")
    axs[1].legend()

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
    sns.kdeplot(data=df, x="score", hue="group", ax=axs[2])

    plt.tight_layout()

    if save_dir is not None:
        name = dataset.split("/")[-1].split("_")[0]
        plt.savefig(f"{save_dir}/{name}_{sens_attr}.pdf")

    plt.show()


if __name__ == "__main__":
    app.cli()
