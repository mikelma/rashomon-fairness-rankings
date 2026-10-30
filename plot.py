import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.ticker as mticker
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
    title: Optional[str] = None,
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
    plt.legend(title=r"$\epsilon$")

    if title is not None:
        plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Fairness improvement (\\%)")
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.tight_layout()

    if save:
        name = dataset.split("/")[-1].split(".")[0]
        plt.savefig(f"figures/lp/{name}_{sens_attr}.svg")

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


@app.command
def inestability(
    k: int = 100,
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

    experiments = [
        ("LawSAT", "data/law_data_processed.csv", "score", "sex"),
        ("LawSAT", "data/law_data_processed.csv", "score", "race"),
        ("HDMA", "data/hmda_processed.csv", "score", "sex"),
        ("HDMA", "data/hmda_processed.csv", "score", "race"),
        ("COMPAS", "data/compas-processed.csv", "score", "sex"),
        ("COMPAS", "data/compas-processed.csv", "score", "race"),
        ("SAT", "data/satgpa_processed.csv", "score", "sex"),
        ("Astronauts", "data/astronauts_processed.csv", "score", "sex"),
    ]

    lst_eps = np.linspace(start=0, stop=1, num=1000)

    def _compute_candidates(dataset, score_attr, sens_attr):
        df = pd.read_csv(dataset)

        print("* Number of individuals:", len(df))
        print("* N =", len(df))
        n = len(df)

        # preprocessing
        scores = df[score_attr][:n].values
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = 1 - scores

        ranking = np.argsort(-scores)
        groups = df[sens_attr][:n].values

        y_k = scores[ranking[k - 1]]
        y_k_plus_one = scores[ranking[k]]

        # the scores of individuals not in the top k
        top = ranking[:k]
        not_top = ranking[k:]

        y_top = scores[top]
        y_not_top = scores[not_top]
        cands = []
        for eps in lst_eps:
            mask_not_top = y_k - y_not_top <= eps
            mask_top = y_top - y_k_plus_one <= eps

            g1_top = groups[top]
            g1_not_top = groups[not_top]

            # p \in (0, 1)
            g1_0_1 = (mask_not_top & g1_not_top).sum() + (mask_top & g1_top).sum()
            g0_0_1 = (mask_not_top & ~g1_not_top).sum() + (mask_top & ~g1_top).sum()

            # p = 1
            g1_1 = (~mask_top & g1_top).sum()
            g0_1 = (~mask_top & ~g1_top).sum()

            # p = 0
            g1_0 = (~mask_not_top & g1_not_top).sum()
            g0_0 = (~mask_not_top & ~g1_not_top).sum()

            # cands.append(
            #     dict(
            #         p_0_1=(g0_0_1, g1_0_1),
            #         p_1=(g0_1, g1_1),
            #         p_0=(g0_0, g1_0),
            #     )
            # )
            cands.append(
                (
                    (g0_0_1, g1_0_1),
                    (g0_1, g1_1),
                    (g0_0, g1_0),
                )
            )

        # dims: (num_eps, prob_classes, num_groups)
        cands = np.asarray(cands)
        return cands, len(df), (groups == 0).sum()

    # markers = ["o", "s", "^", "D", "v", "p", "*", "h", "+", "x"]
    # marker_step = max(1, len(lst_eps) // 4)
    # markersize = 10

    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    axs = axs.flat

    for id, (data_name, dataset, score_attr, sens_attr) in enumerate(experiments):
        ax = axs[id]

        # dims: (num_eps, prob_classes, num_groups)
        result, n, num_group_0 = _compute_candidates(dataset, score_attr, sens_attr)
        num_group_1 = n - num_group_0

        labels = [r"$p \in (0, 1)$", r"$p = 1$", r"$p = 0$"]
        for i, label in enumerate(labels):
            group_0 = result[:, i, 0] / num_group_0
            group_1 = result[:, i, 1] / num_group_1
            line = ax.plot(
                lst_eps,
                group_0,
                label=f"{label} (group 0)",
                linewidth=2,
            )
            ax.plot(
                lst_eps,
                group_1,
                linestyle="dashed",
                label=f"{label} (group 1)",
                color=line[0].get_color(),
                linewidth=2,
            )

        ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.spines[["right", "top"]].set_visible(False)

        sens = "gender" if sens_attr == "sex" else "ethnicity"
        ax.set_title(f"{data_name} ({sens})")

        if id == 0:
            ax.legend()
            ax.set_ylabel(r"\% of candidates per group")
            ax.set_xlabel(r"$\epsilon$ (log)")

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app.cli()
