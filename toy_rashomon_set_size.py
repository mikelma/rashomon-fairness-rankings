import tyro
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
import itertools
from functools import partial
import seaborn as sns


@dataclass
class Args:
    seed: int = 42

    k: int = 3
    n: int = 5
    eps: float = 0.1

    num_samples: int = 100
    max_std: float = 10.0
    num_std: int = 10

    annot: bool = False


@partial(jax.jit, static_argnums=(1,))
def generate_rashomon_set(
    all_top_k: jax.Array, n: int, scores: jax.Array, opt_top_k: jax.Array, eps: float
):
    ident = jnp.full(n, False)

    def _compute_margin(top_k):
        top_k = jnp.array(top_k)
        mask_top_k = ident.at[top_k].set(True)
        mask_opt_k = ident.at[opt_top_k].set(True)

        only_opt = mask_opt_k & (~mask_top_k)
        only_pi = mask_top_k & (~mask_opt_k)

        opt_w = jnp.where(only_opt, scores, 0.0)
        pi_w = jnp.where(only_pi, scores, 0.0)

        # ------------------------------------ #
        #  HAU PAPERREAN BUELTA EMANDA DAGO!!  #
        # ------------------------------------ #
        margin = pi_w.sum() - opt_w.sum()
        # ------------------------------------ #

        # print(top_k, mask_opt_k, mask_top_k, margin)

        return margin

    margins = jax.vmap(_compute_margin)(all_top_k)
    return margins <= eps, margins


@partial(jax.jit, static_argnums=(1,))
def rashomon_set_size(
    all_top_k: jax.Array, n: int, eps: float, scores: jax.Array, opt_top_k: jax.Array
):
    rset, margins = generate_rashomon_set(
        all_top_k=all_top_k,
        n=n,
        scores=scores,
        opt_top_k=opt_top_k,
        eps=eps,
    )
    return rset.sum(), margins


def rset_sizes_from_margins(
    margins: jax.Array, eps_lst: jax.Array, num_top_k: int, num_stds: int
):
    """
    Args:
    -----
    - margins: a (num_samples, num_top_k) shaped array
    - eps_lst: list of different epsilon values to try

    Returns:
    --------
    A tensor (eps_lst.shape[0], num_std, num_top_k) where
    each ijk value counts the number of rashomon sets of size k for
    std j and epsilon i.
    """

    def compute_sizes(eps):
        rset_sizes = (margins <= eps).sum(-1)

        # compute a matrix where each ij value is: for a variance i which
        # is the number of times a rashomon set of size j appears
        mat = jax.vmap(partial(jnp.bincount, length=num_top_k + 1))(
            rset_sizes.reshape(num_stds, -1)
        )
        mat = mat[:, 1:]
        return mat

    mats = jax.vmap(compute_sizes)(eps_lst)
    return mats


def main():
    args = tyro.cli(Args)

    key = jax.random.key(args.seed)

    _key, key = jax.random.split(key)

    # -------------------------------------------------- #
    # Generate the true scores
    # (1)
    # true_scores = jax.random.normal(_key, args.n)
    # (2)
    # true_scores = jnp.arange(args.n) / args.n
    # (3)
    # true_scores = jnp.ones(args.n)
    # true_scores = true_scores.at[:2].set(0.1)
    # -------------------------------------------------- #
    scores = jax.random.normal(_key, (args.num_samples, args.n))
    assert args.num_samples % args.num_std == 0, (
        "The number samples must be dividable by the number of std values"
    )
    repeat = args.num_samples // args.num_std
    std = jnp.linspace(0, args.max_std, num=args.num_std)
    stds = std.repeat(repeat, -1)
    scores *= stds.reshape(-1, 1)
    scores = jnp.abs(scores)

    optimal_top_ks = jnp.argsort(scores)[:, : args.k]

    all_top_k = jnp.asarray(list(itertools.combinations(range(args.n), args.k)))
    num_top_k = all_top_k.shape[0]
    print("Total number of top-k vectors:", num_top_k)

    rset_size, margins = jax.vmap(
        partial(
            rashomon_set_size,
            all_top_k,
            args.n,
            args.eps,
        )
    )(scores, optimal_top_ks)

    # ---------------------------------------------- #
    eps_lst = jnp.linspace(0.001, 1, num=20)
    # dims: (eps_lst.shape[0], num_std, num_top_k)
    sizes = rset_sizes_from_margins(
        margins=margins, eps_lst=eps_lst, num_top_k=num_top_k, num_stds=args.num_std
    )
    percentages = sizes / sizes.sum(-1, keepdims=True)
    # dims: (eps_lst.shape, num_std)
    entropy_map = -(percentages * jnp.log(percentages + 1e-8)).sum(-1) / jnp.log(
        percentages.shape[-1]
    )
    ax = plt.gca()
    sns.heatmap(
        entropy_map,
        annot=args.annot,
        cmap="Blues",
        yticklabels=eps_lst,
        xticklabels=std,
        cbar_kws=dict(label="Entropy of the size vector"),
        ax=ax,
    )
    ax.set_xlabel("Variance")
    ax.set_ylabel("Epsilon")

    plt.show()
    quit()
    # ---------------------------------------------- #

    # compute a matrix where each ij value is: for a variance i which
    # is the number of times a rashomon set of size j appears
    mat = jax.vmap(partial(jnp.bincount, length=all_top_k.shape[0] + 1))(
        rset_size.reshape(args.num_std, -1)
    )
    mat = mat[:, 1:]

    fig = plt.figure()
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # mat = mat > 0.
    sns.heatmap(
        mat,
        annot=args.annot,
        cmap="Blues",
        xticklabels=list(range(1, all_top_k.shape[0] + 1)),
        yticklabels=std,
        ax=ax1,
    )
    ax1.set_xlabel("Rashomon set size")
    ax1.set_ylabel("Variance of the score vector")

    probs = mat / mat.sum(-1).reshape(-1, 1) + 1e-7
    entropies = -(probs * jnp.log(probs)).sum(-1) / jnp.log(std.shape[0])

    ax2.plot(std, entropies)
    ax2.set_xlabel("Variance of the score vector")
    ax2.set_ylabel("Entropy of P(R set size)")

    plt.show()


if __name__ == "__main__":
    main()
