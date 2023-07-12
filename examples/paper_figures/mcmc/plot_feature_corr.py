import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import mi_gaussian


def get_2d_correlations(df, up=True):
    tuples = itertools.combinations(df.columns, 2)
    MI = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    for x, y in tuples:
        _x = df[x].to_numpy()
        _y = df[y].to_numpy()
        mask = np.logical_and(~np.isnan(_x), ~np.isnan(_y))
        mask = mask & np.logical_and(~np.isinf(_x), ~np.isinf(_y))
        _x = _x[mask]
        _y = _y[mask]
        mi = np.clip(mi_gaussian(np.vstack([_x, _y]).T), 0, 1.5)
        if up:
            MI.loc[x, y] = mi
        else:
            MI.loc[y, x] = mi
    return MI


def plot_MI(MI, cmap="Blues"):
    """Plot MI matrix."""
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    MI.index = [ALL_LABELS.get(p, p) for p in MI.index]
    MI.columns = [ALL_LABELS.get(p, p) for p in MI.columns]

    sns.heatmap(
        data=MI,
        ax=ax,
        vmin=0,
        vmax=1.5,
        cmap=cmap,
        linewidths=0.0,
        linecolor="k",
        cbar_kws={"label": "MI", "shrink": 0.3},
        xticklabels=True,
        yticklabels=True,
        square=True,
    )
    plt.tight_layout()


if __name__ == "__main__":
    path = Path("../cadpyr_l5")
    mcmc_df = load_chains(pd.read_csv(path / "out/cADpyr_L5/mcmc_df.csv"), base_path=path)
    mcmc_df = mcmc_df[mcmc_df.cost < 3]
    df = pd.read_csv("../ephys_plot/exp_features.csv", index_col=0).reset_index()
    df.loc[0, "index"] = "SearchThresholdCurrent.soma.v.bpo_threshold_current"
    df.loc[1, "index"] = "SearchHoldingCurrent.soma.v.bpo_holding_current"
    df = df.set_index("index")
    features = [
        f
        for f in mcmc_df.features.columns
        if f in df.index
        and not f.startswith("SpikeRec")
        and not f.startswith("APW")
        and not f.startswith("IV")
    ]
    df = df.loc[features].T
    mcmc_df = mcmc_df["features"][features]

    plt.figure(figsize=(4, 3))
    plt.scatter(
        mcmc_df["Step_150.soma.v.mean_frequency"],
        mcmc_df["Step_150.soma.v.inv_second_ISI"],
        s=1,
        marker=".",
        c="k",
        label="mcmc",
        rasterized=True,
    )
    mask = (df["Step_150.soma.v.inv_second_ISI"] < 2) & (df["Step_150.soma.v.mean_frequency"] > 5)
    _df = df[~mask]
    plt.scatter(
        _df["Step_150.soma.v.mean_frequency"],
        _df["Step_150.soma.v.inv_second_ISI"],
        c="b",
        marker=".",
        label="exp.",
    )
    _df = df[mask]
    print(_df["Step_150.soma.v.mean_frequency"], _df["Step_280.soma.v.mean_frequency"])
    plt.scatter(
        _df["Step_150.soma.v.mean_frequency"],
        _df["Step_150.soma.v.inv_second_ISI"],
        c="b",
        marker="+",
        label="exp. outlier",
    )
    plt.xlabel(ALL_LABELS.get("Step_150.soma.v.mean_frequency"))
    plt.ylabel(ALL_LABELS.get("Step_150.soma.v.inv_second_ISI"))
    plt.legend()
    plt.axis([1, 15, 0, 13])
    plt.tight_layout()
    plt.savefig("step_150_freq.pdf")

    plt.figure(figsize=(4, 3))
    plt.scatter(
        mcmc_df["Step_150.soma.v.mean_frequency"],
        mcmc_df["Step_200.soma.v.mean_frequency"],
        s=1,
        c="k",
        marker=".",
        label="mcmc",
        rasterized=True,
    )
    _df = df[~mask]
    plt.scatter(
        _df["Step_150.soma.v.mean_frequency"],
        _df["Step_200.soma.v.mean_frequency"],
        c="b",
        marker=".",
        label="exp.",
    )
    _df = df[mask]
    plt.scatter(
        _df["Step_150.soma.v.mean_frequency"],
        _df["Step_200.soma.v.mean_frequency"],
        c="b",
        marker="+",
        label="exp. outlier",
    )
    plt.xlabel(ALL_LABELS.get("Step_150.soma.v.mean_frequency"))
    plt.ylabel(ALL_LABELS.get("Step_200.soma.v.mean_frequency"))
    plt.legend()
    plt.axis([1, 15, 4, 18])
    plt.tight_layout()
    plt.savefig("step_150_200_freq.pdf")

    plt.figure(figsize=(4, 3))
    plt.scatter(
        mcmc_df["Step_150.soma.v.mean_frequency"],
        mcmc_df["Step_280.soma.v.mean_frequency"],
        s=1,
        c="k",
        marker=".",
        label="mcmc",
        rasterized=True,
    )
    _df = df[~mask]
    plt.scatter(
        _df["Step_150.soma.v.mean_frequency"],
        _df["Step_280.soma.v.mean_frequency"],
        c="b",
        marker=".",
        label="exp.",
    )
    _df = df[mask]
    plt.scatter(
        _df["Step_150.soma.v.mean_frequency"],
        _df["Step_280.soma.v.mean_frequency"],
        c="b",
        marker="+",
        label="exp. outlier",
    )
    plt.xlabel(ALL_LABELS.get("Step_150.soma.v.mean_frequency"))
    plt.ylabel(ALL_LABELS.get("Step_280.soma.v.mean_frequency"))
    plt.legend()
    plt.axis([1, 15, 7, 23])
    plt.tight_layout()
    plt.savefig("step_150_280_freq.pdf")

    df = df[~mask]
    cor = get_2d_correlations(df, up=True)
    print(cor.loc["Step_150.soma.v.mean_frequency"]["Step_280.soma.v.mean_frequency"])
    mcmc_cor = get_2d_correlations(mcmc_df, up=False)
    print(mcmc_cor.loc["Step_280.soma.v.mean_frequency"]["Step_150.soma.v.mean_frequency"])
    plot_MI(cor, cmap="Blues")
    plt.savefig("exp_feature_correlation.pdf")
    plot_MI(mcmc_cor, cmap="Greys")
    plt.savefig("mcmc_feature_correlation.pdf")
