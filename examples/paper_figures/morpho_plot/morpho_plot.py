from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neurom import NeuriteType
from neurom import get
from neurom import load_morphology
from neurom import view

if __name__ == "__main__":
    path = Path("../cadpyr_l5/")

    large = "rat_P16_S1_RH3_20140129"
    small = "C270999B-P3"
    mean = "vd100617_idB"
    exemplar = "vd110530_idC"

    morph_df = pd.read_csv(path / "out/cADpyr_L5/rediametrized_combo_df.csv")
    morph_df = morph_df[morph_df.mtype == "L5_TPC:A"]
    morph_df = morph_df[["name", "path"]].set_index("name")
    morph_df["path"] = str(path) + "/" + morph_df["path"]

    plt.figure()
    ax = plt.gca()
    for i, t in enumerate(["small", "exemplar", "mean", "large"]):
        m = load_morphology(morph_df.loc[eval(t), "path"])
        m = m.transform(lambda x: x + (i - 1) * np.array([500, 0, 0]))
        view.plot_morph(
            m, ax=ax, neurite_type=NeuriteType.apical_dendrite, realistic_diameters=True
        )
        view.plot_morph(m, ax=ax, neurite_type=NeuriteType.basal_dendrite, realistic_diameters=True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("morphs.png")  # large in pdf

    for gid in morph_df.index:
        m = load_morphology(morph_df.loc[gid, "path"])
        morph_df.loc[gid, "apical surface area"] = get(
            "total_area", m, neurite_type=NeuriteType.apical_dendrite
        )
        morph_df.loc[gid, "basal surface area"] = get(
            "total_area", m, neurite_type=NeuriteType.basal_dendrite
        )
        morph_df.loc[gid, "total surface area"] = (
            morph_df.loc[gid, "apical surface area"] + morph_df.loc[gid, "basal surface area"]
        )
        morph_df.loc[gid, "apical total length"] = get(
            "total_length", m, neurite_type=NeuriteType.apical_dendrite
        )
        morph_df.loc[gid, "basal total length"] = get(
            "total_length", m, neurite_type=NeuriteType.basal_dendrite
        )
        morph_df.loc[gid, "number of bifurcations"] = get(
            "number_of_bifurcations", m, neurite_type=NeuriteType.basal_dendrite
        )

    morph_df = morph_df.drop(columns=["path"])
    morph_df = morph_df.sort_values(by="total surface area")
    morph_df.to_csv("data.csv")

    fig, axs = plt.subplots(1, len(morph_df.columns), figsize=(1.5 * len(morph_df.columns), 3))
    for ax, feat in zip(axs, morph_df.columns):
        d = morph_df[feat]
        sns.stripplot(data=d, ax=ax, color="k")
        sns.stripplot(data=d.loc[[large]], ax=ax, color="r", size=8)
        sns.stripplot(data=d.loc[[small]], ax=ax, color="g", size=8)
        sns.stripplot(data=d.loc[[mean]], ax=ax, color="b", size=8)
        sns.stripplot(data=d.loc[[exemplar]], ax=ax, color="m", size=5)
        ax.axhline(d.mean(), c="k")
        ax.axhline(d.mean() + d.std(), c="k", ls="--")
        ax.axhline(d.mean() - d.std(), c="k", ls="--")
        ax.axhline(d.mean() + 2 * d.std(), c="k", ls="-.")
        ax.axhline(d.mean() - 2 * d.std(), c="k", ls="-.")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_ylabel(feat)
        ax.xaxis.set_ticklabels([])
    plt.tight_layout()
    plt.savefig("morphs_feat.pdf")
