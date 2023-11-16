"""Module to select valid pairs of morphologies and electrical models."""

# Copyright (c) 2022 EPFL-BBP, All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode
# or send a letter to Creative Commons, 171
# Second Street, Suite 300, San Francisco, California, 94105, USA.

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib.backends.backend_pdf import PdfPages

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.utils import FEATURE_FILTER


def select_model_morphos(
    df, clip=5, morpho_thresh=0.2, emodel_thresh=0.05, filters=None, pdf=None, select_folder=None
):
    """Select models and morphologies."""
    if filters is None:
        filters = FEATURE_FILTER

    score_df = df["scores"].apply(
        lambda json_str: pd.Series(json.loads(json_str))
        if isinstance(json_str, str)
        else pd.Series(dtype=float)
    )
    if filters is not None:
        score_df = score_df.drop(
            columns=[c for c in score_df.columns if any(c.startswith(f) for f in filters)]
        )
    df.loc[:, "cost"] = score_df.max(1).to_list()

    # ensure rheobase is shown first if it fails
    df.loc[score_df["SearchThresholdCurrent.soma.v.bpo_threshold_current"] > clip, "cost"] = 1000
    score_df.loc[
        score_df["SearchThresholdCurrent.soma.v.bpo_threshold_current"] > clip,
        "SearchThresholdCurrent.soma.v.bpo_threshold_current",
    ] = 1000

    plt.figure(figsize=(8, 15))
    ax = plt.gca()
    score_df[df.cost > clip].idxmax(1).value_counts(normalize=True, ascending=True).plot.barh(ax=ax)
    plt.xlabel("frac of bad feature for failed cells")
    plt.tight_layout()
    plt.savefig(select_folder / "bad_f.pdf")

    select_df = df.pivot(index="name", columns="emodel", values="cost")
    n_morphs = len(select_df.index)
    n_emodels = len(select_df.columns)

    fail = select_df > 10
    select_df[select_df <= clip] = 0
    select_df[select_df > clip] = 1
    select_df[select_df.isna()] = 1
    select_df[fail] = 2

    morph_select_df = select_df.sum(1).sort_values(ascending=False)
    _morpho_thresh = morpho_thresh * n_emodels
    valid_morphos = morph_select_df.loc[morph_select_df < _morpho_thresh].index

    all_emodel_select_df = select_df.sum(0).sort_values(ascending=False)
    emodel_select_df = select_df.loc[valid_morphos].sum(0).sort_values(ascending=False)
    _emodel_thresh = emodel_thresh * n_morphs
    valid_emodels = emodel_select_df.loc[emodel_select_df < _emodel_thresh].index

    _, ((ax_top, _ax), (ax, ax_right)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(4, 3),
        gridspec_kw={
            "hspace": 0.0,
            "wspace": 0.0,
            "width_ratios": [0.85, 0.15],
            "height_ratios": [0.15, 0.85],
        },
    )
    select_df = select_df.loc[morph_select_df.index, all_emodel_select_df.index]
    ax.imshow(
        select_df.to_numpy(),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Greys",
        extent=[0, n_emodels, 0, n_morphs],
    )
    plot_paper = False  # True
    if plot_paper:
        ax.vlines(15 - 0.5, 0, 5, color="r")
        print("example emodel: ", select_df.columns[15])
        large = "rat_P16_S1_RH3_20140129"
        small = "C270999B-P3"
        mean = "vd100617_idB"
        exemplar = "vd110530_idC"
        ax.hlines(np.argwhere(select_df.index == large) - 0.5, 0, 5, color="b")
        ax.hlines(np.argwhere(select_df.index == small) - 0.5, 0, 5, color="g")
        ax.hlines(np.argwhere(select_df.index == mean) - 0.5, 0, 5, color="r")
        ax.hlines(np.argwhere(select_df.index == exemplar) - 0.5, 0, 5, color="m")

    ax.set_ylabel("morphologies")
    ax.set_xlabel("models")
    ax.set_xlim(0, n_emodels)
    ax.set_ylim(0, n_morphs)
    ax.vlines(
        n_emodels - len(valid_emodels),
        n_morphs - len(valid_morphos),
        n_morphs,
        color="coral",
        path_effects=[patheffects.withTickedStroke(spacing=8, length=1.0, linewidth=0.5)],
    )
    ax.hlines(
        n_morphs - len(valid_morphos),
        n_emodels - len(valid_emodels),
        n_emodels,
        color="coral",
        path_effects=[
            patheffects.withTickedStroke(spacing=8, length=1.0, linewidth=0.5, angle=-135)
        ],
    )

    ax_right.barh(range(len(morph_select_df)), morph_select_df.to_numpy(), color="k")
    ax_right.axhline(
        n_morphs - len(valid_morphos) - 0.5,
        c="coral",
        path_effects=[
            patheffects.withTickedStroke(spacing=8, length=1.0, linewidth=0.5, angle=-135)
        ],
    )
    ax_right.set_xlabel("# failed")
    ax_right.set_ylim(-0.5, n_morphs - 0.5)
    ax_right.set_yticks([])
    ax_right.set_yticklabels([])
    ax_right.spines.right.set_visible(False)
    ax_right.spines.bottom.set_visible(False)
    ax_right.spines.top.set_visible(False)

    ax_top.bar(range(n_emodels), all_emodel_select_df.to_numpy(), color="k")
    ax_top.axvline(
        n_emodels - len(valid_emodels) - 0.5,
        c="coral",
        path_effects=[patheffects.withTickedStroke(spacing=8, length=1.0, linewidth=0.5)],
    )
    ax_top.set_ylabel("# failed")
    ax_top.set_xlim(-0.5, n_emodels - 0.5)
    ax_top.set_xticks([])
    ax_top.set_xticklabels([])
    ax_top.spines.right.set_visible(False)
    ax_top.spines.left.set_visible(False)
    ax_top.spines.top.set_visible(False)

    _ax.set_frame_on(False)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_yticklabels([])
    _ax.set_xticklabels([])

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    return valid_morphos, valid_emodels, select_df


def plot_select_corner(_df, pdf, df_params, valid_emodels):
    """Corner plot with scatter of valid models."""
    emodels = df_params["emodel"]
    df_params = df_params["normalized_parameters"]
    df_params["emodel"] = emodels
    _df = df_params[df_params.emodel.isin(_df.emodel.unique())]
    _df = _df.set_index("emodel")
    _df = _df.melt(ignore_index=False)
    _df["valid"] = False
    _df.loc[valid_emodels, "valid"] = True
    _df["variable"] = [ALL_LABELS[p] for p in _df["variable"]]

    params = _df.variable.unique()
    n_params = len(params)
    fig = plt.figure(figsize=(0.5 * n_params, 0.5 * n_params))
    gs = fig.add_gridspec(n_params, n_params, hspace=0.1, wspace=0.1)
    for i, param1 in enumerate(params):
        _param1 = ALL_LABELS.get(param1, param1)
        for j, param2 in enumerate(params):
            _param2 = ALL_LABELS.get(param2, param2)
            ax = plt.subplot(gs[i, j])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            if j >= i:
                ax.set_frame_on(False)
            else:
                __df = _df[_df.valid]
                plt.scatter(
                    __df.loc[__df.variable == param2, "value"],
                    __df.loc[__df.variable == param1, "value"],
                    s=0.5,
                    c="C1",
                )
                __df = _df[~_df.valid]
                plt.scatter(
                    __df.loc[__df.variable == param2, "value"],
                    __df.loc[__df.variable == param1, "value"],
                    s=0.5,
                    c="C0",
                )

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            if j == 0:
                ax.set_ylabel(_param1, rotation="horizontal", horizontalalignment="right")
            if i == n_params - 1:
                ax.set_xlabel(_param2, rotation="vertical")
            if j == 0 and i == 0:
                ax.set_ylabel(_param1, rotation="horizontal", horizontalalignment="left")
            if i == j + 1:
                ax.set_ylabel(_param1, rotation="horizontal", horizontalalignment="left")
                ax.yaxis.set_label_position("right")

    pdf.savefig()
    plt.close()


def plot_select_stripplot(_df, pdf, df_params, valid_emodels):
    """Stripplot of selected emodels."""
    plt.figure(figsize=(6, 20))
    emodels = df_params["emodel"]
    df_params = df_params["normalized_parameters"]
    df_params["emodel"] = emodels
    _df = df_params[df_params.emodel.isin(_df.emodel.unique())]
    _df = _df.set_index("emodel")
    _df = _df.melt(ignore_index=False)
    _df["valid"] = False
    _df.loc[valid_emodels, "valid"] = True
    _df["variable"] = [ALL_LABELS[p] for p in _df["variable"]]

    sns.stripplot(data=_df, x="value", y="variable", hue="valid", ax=plt.gca(), dodge=True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


def plot_feature_distributions(df, efeatures, select_folder):
    """Plot feature distributions."""
    with PdfPages(select_folder / "feature_distributions.pdf") as _pdf:
        feat_df = df["features"].apply(
            lambda json_str: pd.Series(json.loads(json_str))
            if isinstance(json_str, str)
            else pd.Series(dtype=float)
        )
        for feat in feat_df.columns:
            feat_split = feat.split(".")
            if feat_split[0] == "RinProtocol":
                feat_split[0] = "Rin"
            if feat_split[0] == "RMPProtocol":
                feat_split[0] = "RMP"
            if feat_split[0] == "SearchHoldingCurrent":
                feat_split[0] = "RinHoldCurrent"
            if feat_split[0] == "SearchThresholdCurrent":
                feat_split[0] = "Threshold"
            for f in efeatures[feat_split[0]][".".join(feat_split[1:3])]:
                if f["feature"] == feat_split[3]:
                    f_mean = f["val"][0]
                    f_std = f["val"][1]

            plt.figure(figsize=(5, 3))
            plt.hist(
                np.clip(feat_df[feat], f_mean - 5 * f_std, f_mean + 5 * f_std),
                bins=100,
                histtype="step",
            )
            plt.axvline(f_mean, c="k")
            plt.axvline(f_mean - f_std, c="k", ls="--")
            plt.axvline(f_mean + f_std, c="k", ls="--")
            plt.axvline(f_mean - 2 * f_std, c="k", ls="-.")
            plt.axvline(f_mean + 2 * f_std, c="k", ls="-.")
            plt.axvline(f_mean - 5 * f_std, c="k", ls="dotted")
            plt.axvline(f_mean + 5 * f_std, c="k", ls="dotted")
            plt.xlabel(feat)
            plt.ylabel("# combos")
            plt.tight_layout()
            _pdf.savefig()
            plt.close()


def select_valid(
    df, emodel, select_folder, morpho_thresh, emodel_thresh, clip, access_point, mtype
):
    """Select valid and plot."""
    selected = {"emodels": {}, "morphos": {}, "stats": {"morphos": {}, "emodels": {}}}
    with PdfPages(select_folder / f"select_{mtype}.pdf") as pdf:
        valid_morphos, valid_emodels, select_df = select_model_morphos(
            df,
            pdf=pdf,
            morpho_thresh=morpho_thresh,
            emodel_thresh=emodel_thresh,
            clip=clip,
            select_folder=select_folder,
        )
        select_df.to_csv(select_folder / f"select_df_{mtype}.csv")

        frac_emodel = len(valid_emodels) / len(select_df.columns)
        frac_morphos = len(valid_morphos) / len(select_df.index)
        print(f"frac emodel: {frac_emodel}, frac morphos: {frac_morphos}")

        selected["emodels"] = valid_emodels.to_list()
        selected["morphos"] = valid_morphos.to_list()
        selected["stats"]["morphos"] = frac_morphos
        selected["stats"]["emodels"] = frac_emodel

    efeatures = access_point.get_json(emodel, "features")
    plot_feature_distributions(df, efeatures, select_folder)
    return selected
