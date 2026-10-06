import tyro
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
import itertools
from functools import partial


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

        # ------------------------------------ #
        #  HAU PAPERREAN BUELTA EMANDA DAGO!!  #
        # ------------------------------------ #
        margin = pi_w.sum() - opt_w.sum()
        # ------------------------------------ #

        # print(top_k, mask_opt_k, mask_top_k, margin)

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
            # jax.debug.print(
            #     "rset: {r}\nmargs: {m}\nscores: {s}", r=rset, m=margs, s=scores
            # )
            return rset.sum(-1)

        return jax.vmap(map_over_std)(batched_scores, opt_top_k)

    return jax.vmap(map_over_eps)(batched_eps)


# @partial(jax.jit, static_argnums=(1,))
# def rashomon_set_size(
#     all_top_k: jax.Array, n: int, eps: float, scores: jax.Array, opt_top_k: jax.Array
# ):
#     rset, margins = generate_rashomon_set(
#         all_top_k=all_top_k,
#         n=n,
#         scores=scores,
#         opt_top_k=opt_top_k,
#         eps=eps,
#     )
#     return rset.sum(), margins


# def batched_rset_sizes(
#     batched_scores: jax.Array, batched_eps: jax.Array, optimal_top_ks: jax.Array, n: int
# ):
#     def _size_for_eps(eps):
#         rset_size, _margins = jax.vmap(
#             partial(
#                 rashomon_set_size,
#                 optimal_top_ks,
#                 n,
#                 eps,
#             )
#         )(batched_scores, optimal_top_ks)
#         return rset_size

#     batched_sizes = jax.vmap(_size_for_eps)(batched_eps)
#     return batched_sizes


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
    scores = jax.random.normal(_key, (args.num_std, args.num_samples, args.n))
    # scores = jnp.abs(scores)
    std = jnp.linspace(0, args.max_std, num=args.num_std)
    scores *= std.reshape(-1, 1, 1)
    # dims: (num_std, num_samples, K)
    optimal_top_ks = jnp.argsort(scores, axis=-1)[:, :, : args.k]

    print("scores:", scores.shape)
    print("optimal_top_ks:", optimal_top_ks.shape)

    all_top_k = jnp.asarray(list(itertools.combinations(range(args.n), args.k)))
    num_top_k = all_top_k.shape[0]
    print("Total number of top-k vectors:", num_top_k)

    batched_epsilon = jnp.linspace(
        args.eps_range[0], args.eps_range[1], num=args.num_eps
    )
    print("batched_epsilon:", batched_epsilon.shape)

    # dims: (num_eps, num_std, num_samples)
    # batched_sizes = batched_rset_sizes(
    #     scores, batched_epsilon, optimal_top_ks, all_top_k, args.n
    # )

    batched_sizes = batched_rset_sizes(
        batched_scores=scores,
        batched_eps=batched_epsilon,
        opt_top_k=optimal_top_ks,
        all_top_ks=all_top_k,
        n=args.n,
    )
    # print(batched_sizes.shape)
    # quit()

    # # dims: (num_eps, num_samples)
    # batched_sizes = batched_rset_sizes(
    #     batched_scores=scores,
    #     batched_eps=batched_epsilon,
    #     optimal_top_ks=optimal_top_ks,
    #     n=args.n,
    # )
    # # dims (num_eps, num_samples / num_std, num_std)
    # batched_sizes = batched_sizes.reshape(batched_epsilon.shape[0], args.num_std, -1)

    batched_means = batched_sizes.mean(-1)
    batched_stds = batched_sizes.std(-1)

    matplotlib.rcParams.update({"font.size": 16})

    for i in range(args.num_eps):
        mean = batched_means[i]
        std_dev = batched_stds[i]
        eps = batched_epsilon[i]
        print(mean.shape, std_dev.shape)

        plt.plot(std, mean, label=f"{round(float(eps), 3)}")
        plt.fill_between(
            std,
            mean - std_dev,
            mean + std_dev,
            alpha=0.3,
        )

    plt.legend(title="Epsilon")
    plt.axhline(y=1, color="tab:grey", linestyle="dashed")
    plt.xlabel("Standard deviation of the score distribution")
    plt.ylabel("Size of the reshomon set")
    plt.show()


if __name__ == "__main__":
    main()
