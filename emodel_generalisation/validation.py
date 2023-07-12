""""Module for Validation of electrical models."""

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from emodel_generalisation.exemplars import get_bins
from emodel_generalisation.exemplars import get_surface_profile

# from emodel_generalisation.extra.features import get_features
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_scores


def validate_morphologies(df, selected, folder, select_folder, exemplar_data, mtype):
    """Validate morphologies with plots."""
    bin_params = {"min": 0, "max": 500, "n": 50}
    path_bins = get_bins(bin_params)
    df = df[df.mtype == mtype].drop_duplicates(subset="name").reset_index()

    surf_df_basal = get_surface_profile(df, path_bins, "basal", morphology_path="path")
    surf_df_apical = get_surface_profile(df, path_bins, "apical", morphology_path="path")
    surf_df_basal.index = df.name
    surf_df_apical.index = df.name

    with PdfPages(folder / f"validation_{mtype}.pdf") as pdf:
        plt.figure(figsize=(4, 3))

        select_df = pd.read_csv(select_folder / f"select_df_{mtype}.csv", index_col=0)
        select_df = select_df.sum(1)
        select_df = select_df.loc[surf_df_basal.index]  # * 1.5 + 10
        select_df.to_csv(f"select_scatter_{mtype}.csv")
        plt.scatter(
            surf_df_basal.loc[selected["morphos"]].sum(1)
            + surf_df_apical.loc[selected["morphos"]].sum(1),
            select_df.loc[selected["morphos"]],
            c="k",
            label="selected",
        )
        plt.scatter(
            surf_df_basal.drop(selected["morphos"]).sum(1)
            + surf_df_apical.drop(selected["morphos"]).sum(1),
            select_df.drop(selected["morphos"]),
            c="coral",
            label="failed",
        )
        exemplar_name = Path(exemplar_data["paths"][mtype]).stem
        plt.scatter(
            surf_df_basal.loc[exemplar_name].sum() + surf_df_apical.loc[exemplar_name].sum(),
            select_df.loc[exemplar_name],
            c="m",
            marker="+",
            label="exemplar",
            s=100,
        )
        plot_paper = True
        if plot_paper:
            try:
                large = "rat_P16_S1_RH3_20140129"
                plt.scatter(
                    surf_df_basal.loc[large].sum() + surf_df_apical.loc[large].sum(),
                    select_df.loc[large],
                    c="b",
                    marker="+",
                    label="large",
                    s=100,
                )
                small = "C270999B-P3"
                plt.scatter(
                    surf_df_basal.loc[small].sum() + surf_df_apical.loc[small].sum(),
                    select_df.loc[small],
                    c="g",
                    label="small",
                    marker="+",
                    s=100,
                )
                mean = "vd100617_idB"
                plt.scatter(
                    surf_df_basal.loc[mean].sum() + surf_df_apical.loc[mean].sum(),
                    select_df.loc[mean],
                    c="r",
                    label="mean",
                    marker="+",
                    s=100,
                )
            except Exception:  # pylint:disable=broad-exception-caught
                pass

        plt.legend()
        plt.xlabel("proximal surface area")
        plt.ylabel("# failed models")
        plt.tight_layout()
        pdf.savefig()

        plt.close()
        plt.figure()

        for _n in surf_df_basal.index:
            c = "k"
            if _n in selected["morphos"]:
                c = "r"
            if _n == exemplar_name:
                c = "b"

            plt.plot(surf_df_basal.columns, surf_df_basal.loc[_n], c=c, lw=0.5)
        plt.xlabel("distance to soma")
        plt.ylabel("basal surface area")
        pdf.savefig()
        plt.close()

        plt.figure()
        for _n in surf_df_apical.index:
            c = "k"
            if _n in selected["morphos"]:
                c = "r"
            if _n == exemplar_name:
                c = "b"

            plt.plot(surf_df_apical.columns, surf_df_apical.loc[_n], c=c, lw=0.5)
        plt.xlabel("distance to soma")
        plt.ylabel("apical surface area")
        pdf.savefig()
        plt.close()

        plt.figure()
        for _n in surf_df_apical.index:
            c = "k"
            if _n in selected["morphos"]:
                c = "r"
            if _n == exemplar_name:
                c = "b"

            plt.plot(
                surf_df_apical.columns,
                surf_df_apical.loc[_n] + surf_df_basal.loc[_n],
                c=c,
                lw=0.5,
            )
        pdf.savefig()
        plt.close()


def _filter(df):
    df = get_scores(df.copy())
    score_df = df["scores_raw"].apply(
        lambda json_str: pd.Series(json.loads(json_str))
        if isinstance(json_str, str)
        else pd.Series(dtype=float)
    )
    score_df = score_df.drop(
        columns=[c for c in score_df.columns if any(c.startswith(f) for f in FEATURE_FILTER)]
    )

    df.loc[:, "cost"] = score_df.max(1).to_list()
    return df


def plot_adaptation_summaries(df, df_no, filename="adaptation_summary.pdf"):
    """Plot adaptation summaries."""
    with PdfPages(filename) as pdf:
        plt.figure(figsize=(5, 3))

        plt.hist(df["ais_scaler"], bins=40, histtype="step", label="AIS scaler", log=True)
        plt.hist(df["soma_scaler"], bins=40, histtype="step", label="soma scaler", log=True)
        plt.axvline(1, c="k")
        plt.legend()
        plt.xlabel("scaler")
        plt.ylabel("# models")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(5, 3))
        plt.hist(df["rin_no_ais"], bins=40, histtype="step", label="Rin no AIS", log=True)
        plt.hist(df["rin_no_soma"], bins=40, histtype="step", label="Rin no soma", log=True)
        plt.axvline(1, c="k")
        plt.legend()
        plt.xlabel("Rin")
        plt.ylabel("# models")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(5, 3))

        df = _filter(df)
        df_no = _filter(df_no)

        df = df.set_index(["name", "emodel"])
        df_no = df_no.set_index(["name", "emodel"])
        df_no = df_no.loc[df.index]
        df_no["cost"] = np.clip(df_no["cost"], 0, 20)
        df["cost"] = np.clip(df["cost"], 0, 20)
        plt.hist(
            df["cost"] - df_no["cost"],
            histtype="step",
            bins=40,
            log=True,
            density=True,
            color="k",
        )

        a = df["cost"] - df_no["cost"]
        print("frac improve:", len(a[a < 0]) / len(a))
        print("frac bad :", len(a[a > 5]) / len(a))
        print("frac fix:", len(a[a < -5]) / len(a))
        plt.legend()
        plt.tight_layout()
        plt.xlabel("cost adapted - not adapted")
        plt.ylabel("density of models")
        plt.axvline(-5, c="k", ls="--")
        plt.axvline(5, c="k", ls="--")
        plt.tight_layout()
        pdf.savefig()
        plt.close()


def compare_adaptation(df_adapted, df_not_adapted, clip, out_path):
    "Commare adapted and not adapted." ""
    df_not_adapted = df_not_adapted[df_not_adapted.emodel.isin(df_adapted.emodel.unique())]
    df_adapted = _filter(df_adapted)
    df_not_adapted = _filter(df_not_adapted)

    plt.figure()
    df_adapted = df_adapted.set_index(["name", "emodel"])
    df_not_adapted = df_not_adapted.set_index(["name", "emodel"])
    df_not_adapted = df_not_adapted.loc[df_adapted.index]

    df_adapted["cost_not_adapted"] = np.clip(df_not_adapted["cost"], 0, clip)
    df_adapted["cost_adapted"] = np.clip(df_adapted["cost"], 0, clip)

    _, ((ax_top, _ax), (ax, ax_right)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(6, 5),
        gridspec_kw={
            "hspace": 0.0,
            "wspace": 0.0,
            "width_ratios": [0.8, 0.2],
            "height_ratios": [0.2, 0.8],
        },
    )

    ax.scatter(
        df_adapted["cost_adapted"],
        df_adapted["cost_not_adapted"],
        s=0.1,
        marker=".",
    )

    ax_right.hist(
        df_adapted["cost_not_adapted"],
        range=[2.3, clip + 0.1],
        histtype="step",
        orientation="horizontal",
        bins=50,
    )
    ax_top.hist(
        df_adapted["cost_adapted"],
        range=[2.3, clip + 0.1],
        histtype="step",
        bins=50,
    )
    ax.plot([0, clip + 1], [0, clip + 1], c="k", ls="--")
    ax.set_ylabel("cost not adapted")
    ax.set_xlabel("cost adapted")
    ax.set_xlim(2.3, clip + 0.1)
    ax.set_ylim(2.3, clip + 0.1)
    ax.legend()

    ax_right.set_yticks([])
    ax_right.set_yticklabels([])
    y_lim = max(ax_top.get_ylim()[1], ax_right.get_xlim()[1])
    ax_right.set_xlim(0, y_lim)
    ax_right.set_ylim(2.3, clip + 0.1)

    ax_top.set_xticks([])
    ax_top.set_xticklabels([])
    ax_top.set_ylim(0, y_lim)
    ax_top.set_xlim(2.3, clip + 0.1)

    _ax.set_frame_on(False)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_yticklabels([])
    _ax.set_xticklabels([])

    plt.savefig(out_path)
