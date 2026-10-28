import tyro
import jax
import jax.numpy as jnp
from jax.scipy.special import comb
from jax.scipy.stats.norm import ppf as normal_ppf
from jax.scipy.stats.norm import pdf as normal_pdf
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
import itertools
from functools import partial
from typing import Optional


@dataclass
class Args:
    seed: int = 42

    k: int = 3
    n: int = 5

    num_samples: int = 100

    eps_range: list[float] = field(default_factory=lambda: [0.0, 1.0])
    num_eps: int = 5

    max_std: float = 10.0
    num_std: int = 10

    approx_bounds: bool = False
    normal_bounds: bool = False

    font_size: int = 16
    save: Optional[str] = None


@partial(jax.jit, static_argnums=(1,))
def generate_rashomon_set(
    all_top_k: jax.Array, n: int, scores: jax.Array, opt_top_k: jax.Array, eps: float
):
    ident = jnp.full(n, False)

    def _compute_margin(top_k):
        # top_k = jnp.array(top_k)
        mask_top_k = ident.at[top_k].set(True)
        mask_opt_k = ident.at[opt_top_k].set(True)

        only_opt = mask_opt_k & (~mask_top_k)
        only_pi = mask_top_k & (~mask_opt_k)

        opt_w = jnp.where(only_opt, scores, 0.0)
        pi_w = jnp.where(only_pi, scores, 0.0)

        margin = opt_w.sum() - pi_w.sum()

        return margin

    margins = jax.vmap(_compute_margin)(all_top_k)
    return margins <= eps, margins


def batched_rset_sizes(
    batched_scores: jax.Array,
    batched_eps: jax.Array,
    opt_top_k: jax.Array,
    all_top_ks: jax.Array,
    n: int,
):
    def map_over_eps(eps):
        def map_over_std(scores, opt_k):
            # dims: (num_samples, num_top_ks)
            rset, margs = jax.vmap(
                generate_rashomon_set, in_axes=(None, None, 0, 0, None)
            )(all_top_ks, n, scores, opt_k, eps)
            return rset.sum(-1)

        return jax.vmap(map_over_std)(batched_scores, opt_top_k)

    return jax.vmap(map_over_eps)(batched_eps)


def size_bound(scores: jax.Array, eps: float, k: int):
    n = scores.shape[0]

    rank = jnp.argsort(-scores)  # Bayes ranking

    gamma = scores[rank[k - 1]] - scores[rank[k]]

    # when gamma == 0 we can change all K items
    M = jnp.where(gamma == 0.0, float(k), jnp.minimum(jnp.floor(eps / gamma), k))

    def _combs(m):
        return comb(k, m) * comb(n - k, m)

    kk = jnp.arange(k + 1)
    vv = jax.vmap(_combs)(kk)
    result = jnp.where(kk <= M, vv, 0).sum()

    return result


def size_normal_bound(n: int, var: float, eps: float, k: int):
    def _combs(m):
        frac = (n * eps) / (m * var)
        rexp = 1 - jnp.exp(
            -frac * normal_pdf(normal_ppf(1 - k / n, scale=var), scale=var)
        )
        return comb(k, m) * comb(n - k, m) * rexp

    kk = jnp.arange(1, k + 1)
    vv = jax.vmap(_combs)(kk)

    stop = jnp.minimum(k, n - k)
    # TODO is this < or <= ?
    result = 1 + jnp.where(kk < stop, vv, 0).sum()

    return result


def main():
    args = tyro.cli(Args)

    key = jax.random.key(args.seed)

    _key, key = jax.random.split(key)
    scores = jax.random.normal(_key, (args.num_std, args.num_samples, args.n))
    # scores = jnp.abs(scores)
    std = jnp.linspace(0, args.max_std, num=args.num_std)
    scores *= std.reshape(-1, 1, 1)

    # dims: (num_std, num_samples, K)
    optimal_top_ks = jnp.argsort(-scores, axis=-1)[:, :, : args.k]

    all_top_k = jnp.asarray(list(itertools.combinations(range(args.n), args.k)))
    num_top_k = all_top_k.shape[0]
    print("Total number of top-k vectors:", num_top_k)

    batched_epsilon = jnp.linspace(
        args.eps_range[0], args.eps_range[1], num=args.num_eps
    )

    batched_sizes = batched_rset_sizes(
        batched_scores=scores,
        batched_eps=batched_epsilon,
        opt_top_k=optimal_top_ks,
        all_top_ks=all_top_k,
        n=args.n,
    )
    batched_means = batched_sizes.mean(-1)
    batched_stds = batched_sizes.std(-1)

    # compute the theoretical upper bounds
    print("scores:", scores.shape)
    print("epsilons", batched_epsilon.shape)

    def _batched_bound_compute(eps):
        bounds = jax.vmap(size_bound, in_axes=(0, None, None))(
            scores.reshape(-1, args.n), eps, args.k
        )
        bounds = bounds.reshape(*scores.shape[:2])
        return bounds

    def _batched_normal_bound(eps):
        bounds = jax.vmap(size_normal_bound, in_axes=(None, 0, None, None))(
            args.n, std, eps, args.k
        )
        return bounds

    matplotlib.rcParams.update(
        {
            "font.size": args.font_size,
            "font.family": "serif",
            "text.usetex": True,
            "pdf.fonttype": 42,
        }
    )

    for i in range(args.num_eps):
        mean = batched_means[i]
        std_dev = batched_stds[i]
        eps = batched_epsilon[i]

        line = plt.plot(std, mean, label=f"{round(float(eps), 2)}", linewidth=2)
        plt.fill_between(
            std,
            mean - std_dev,
            mean + std_dev,
            alpha=0.3,
        )

        if args.approx_bounds:
            # dims: (num_eps, num_std, num_samples)
            batched_bounds = jax.vmap(_batched_bound_compute)(batched_epsilon)
            # bounds_stds = batched_bounds.std(-1)
            bounds_approx = batched_bounds.mean(-1)
            plt.plot(
                std,
                bounds_approx[i],
                linestyle="dashed",
                color=line[0].get_color(),
                linewidth=2,
            )
        if args.normal_bounds:
            bounds_normal = jax.vmap(_batched_normal_bound)(batched_epsilon)
            plt.plot(
                std,
                bounds_normal[i],
                linestyle="dotted",
                color=line[0].get_color(),
                linewidth=2,
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    if args.approx_bounds:
        # add the upper entry to the legend
        ub_handle = matplotlib.lines.Line2D(
            [], [], color="black", linestyle="dashed", linewidth=1.5, label="UB"
        )
        handles.append(ub_handle)
        labels.append("UB approx")
    if args.normal_bounds:
        # add the upper entry to the legend
        ub_handle = matplotlib.lines.Line2D(
            [], [], color="black", linestyle="dotted", linewidth=1.5, label="UB"
        )
        handles.append(ub_handle)
        labels.append("UB normal")

    # Pass the combined handles and labels to plt.legend
    plt.legend(
        handles=handles,
        labels=labels,
        title=r"$\epsilon$",
        frameon=True,
        fancybox=False,
    )

    # plt.axhline(y=1, color="tab:grey", linestyle="dashed")
    plt.xlabel("Std. dev. of the score distribution")
    plt.ylabel(r"$|\mathcal{R}_\epsilon^K|$")
    # force 1 to be included in the Y-axis tick labels
    locs, labels_y = plt.yticks()
    ticks = [1.0] + [t for t in locs if t > 1.0]
    plt.yticks(ticks)
    plt.ylim(bottom=1)
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.title(f"N={args.n}, K={args.k}")

    if args.save:
        plt.savefig(args.save)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
