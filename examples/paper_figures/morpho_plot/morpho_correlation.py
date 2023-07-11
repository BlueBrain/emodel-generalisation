import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neurom import NeuriteType
from neurom import get
from neurom import load_morphology

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
        # mi = pearsonr(_x, _y)[0]
        if up:
            MI.loc[x, y] = mi
        else:
            MI.loc[y, x] = mi
    return MI


def plot_MI(MI, cmap="Blues"):
    """Plot MI matrix."""
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

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
    df = pd.DataFrame()
    synth_df = pd.DataFrame()

    path = Path("../cadpyr_l5/")
    synth_path = Path("../cadpyr_l5/synthesis_evaluation/")
    morph_df = pd.read_csv(path / "out/cADpyr_L5/rediametrized_combo_df.csv")
    synth_morph_df = pd.read_csv(synth_path / "out/morphs_df/vacuum_synth_morphs_df.csv")
    morph_df = morph_df[morph_df.mtype == "L5_TPC:A"]
    morph_df = morph_df[["name", "path"]].set_index("name")
    morph_df["path"] = str(path) + "/" + morph_df["path"]
    synth_morph_df["path"] = str(synth_path) + "/morphologies_with_ais/" + synth_morph_df["name"]
    feats = {
        "total_area": "surface area",
        "total_length": "total length",
        "number_of_bifurcations": "number of bifurcations",
        "trunk_section_lengths": "mean trunk section lengths",
        "volume_density": "volume_density",
        "total_width": "total width",
        "number_of_sections": "number of sections",
        "number_of_leaves": "number of leaves",
        "section_lengths": "mean section lengths",
        "section_branch_orders": "mean section branch orders",
        "section_term_branch_orders": "mean terminal branch orders",
    }
    for gid in morph_df.index:
        m = load_morphology(morph_df.loc[gid, "path"])
        for f, _f in feats.items():
            df.loc[gid, "basal " + _f] = np.mean(get(f, m, neurite_type=NeuriteType.basal_dendrite))
            df.loc[gid, "apical " + _f] = np.mean(
                get(f, m, neurite_type=NeuriteType.apical_dendrite)
            )

    df = df.sort_index(axis=1)
    cor = get_2d_correlations(df)
    plot_MI(cor, cmap="Blues")
    plt.savefig("morpho_correlations.pdf")
    print(cor)

    for gid in synth_morph_df.index:
        m = load_morphology(synth_morph_df.loc[gid, "path"])
        for f, _f in feats.items():
            synth_df.loc[gid, "basal " + _f] = np.mean(
                get(f, m, neurite_type=NeuriteType.basal_dendrite)
            )
            synth_df.loc[gid, "apical " + _f] = np.mean(
                get(f, m, neurite_type=NeuriteType.apical_dendrite)
            )

    synth_df = synth_df.sort_index(axis=1)
    cor = get_2d_correlations(synth_df)
    plot_MI(cor, cmap="Greys")
    plt.savefig("synth_morpho_correlations.pdf")
    print(cor)

    plt.figure(figsize=(4, 3))
    plt.scatter(
        df["apical surface area"],
        df["apical number of sections"],
        s=10,
        marker=".",
        c="b",
    )
    plt.scatter(
        synth_df["apical surface area"],
        synth_df["apical number of sections"],
        s=10,
        marker=".",
        c="k",
    )
    plt.xlabel("apical surface area")
    plt.ylabel("apical number of sections")
    plt.tight_layout()
    plt.savefig("apical_area_section.pdf")

    plt.figure(figsize=(4, 3))
    plt.scatter(
        df["apical surface area"],
        df["apical mean section lengths"],
        s=10,
        marker=".",
        c="b",
    )
    plt.scatter(
        synth_df["apical surface area"],
        synth_df["apical mean section lengths"],
        s=10,
        marker=".",
        c="k",
    )
    plt.xlabel("apical surface area")
    plt.ylabel("apical mean section lengths")
    plt.tight_layout()
    plt.savefig("apical_area_section_length.pdf")

    plt.figure(figsize=(4, 3))
    plt.scatter(
        df["apical surface area"],
        df["basal surface area"],
        s=10,
        marker=".",
        c="b",
    )
    plt.scatter(
        synth_df["apical surface area"],
        synth_df["basal surface area"],
        s=10,
        marker=".",
        c="k",
    )
    plt.xlabel("apical surface area")
    plt.ylabel("basal surface area")
    plt.tight_layout()
    plt.savefig("apical_area_basal_area.pdf")
