"""Other util functions."""

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
import pickle
from copy import deepcopy
from functools import partial
from hashlib import sha256
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluepyopt.ephys.responses import TimeVoltageResponse
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

FEATURE_FILTER = ["APWaveform", "bAP", "SpikeRec", "IV"]


def cluster_matrix(df, distance=False):
    """Return sorted labels to cluster a matrix with linkage.

    If distance matrix already set distance=True.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        _data = df.to_numpy() if distance else 1.0 / df.to_numpy()

    _data[_data > 1e10] = 1000
    np.fill_diagonal(_data, 0.0)
    dists = squareform(_data)
    Z = linkage(dists, "ward")
    labels = df.columns.to_numpy()
    dn = dendrogram(Z, labels=labels, ax=None)
    return labels[dn["leaves"]]


def _filter_features(combo, features=None, method="ignore"):
    """delete or keep features if method==ignore, resp. method==keep."""
    if isinstance(combo["scores"], dict):
        keys = deepcopy(list(combo["scores"].keys()))
        for key in keys:
            if method == "ignore":
                for feat in features:
                    if feat in key.split("."):
                        del combo["scores"][key]
            elif method == "keep":
                if not any(feat in key.split(".") for feat in features):
                    if key in combo["scores"]:
                        del combo["scores"][key]
    return combo


def get_scores(morphs_combos_df, features_to_ignore=None, features_to_keep=None, clip=250):
    """compute the median and max scores from computations on filtered features."""
    morphs_combos_df.loc[:, "scores_raw"] = morphs_combos_df["scores"]
    morphs_combos_df.loc[:, "scores"] = morphs_combos_df["scores_raw"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and len(s) > 0 else s
    )
    filter_features = None
    if features_to_ignore is not None:
        if features_to_keep is not None:
            raise Exception("please provide only a list of features to ignore or to keep")
        filter_features = partial(_filter_features, features=features_to_ignore, method="ignore")

    if features_to_keep is not None:
        if features_to_ignore is not None:
            raise Exception("please provide only a list of features to ignore or to keep")
        filter_features = partial(_filter_features, features=features_to_keep, method="keep")

    if filter_features is not None:
        morphs_combos_df.apply(filter_features, axis=1)

    morphs_combos_df.loc[:, "median_score"] = morphs_combos_df["scores"].apply(
        lambda score: np.clip(np.median(list(score.values())), 0, clip)
        if isinstance(score, dict)
        else np.nan
    )
    morphs_combos_df.loc[:, "max_score"] = morphs_combos_df["scores"].apply(
        lambda score: np.clip(np.max(list(score.values())), 0, clip)
        if isinstance(score, dict)
        else np.nan
    )
    morphs_combos_df.loc[:, "cost"] = morphs_combos_df["scores"].apply(
        lambda score: np.max(list(score.values())) if isinstance(score, dict) else np.nan
    )
    return morphs_combos_df


def get_feature_df(df, filters=None):
    """Get feature df from complete df."""
    feature_df = df["features"].apply(
        lambda json_str: pd.Series(json.loads(json_str))
        if isinstance(json_str, str)
        else pd.Series(dtype=float)
    )
    if filters is not None:
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if any(c.startswith(f) for f in filters)]
        )
        feature_df = feature_df.drop(
            columns=[c for c in feature_df.columns if any(c.endswith(f) for f in filters)]
        )
    return feature_df


def get_score_df(df, filters=None):
    """Get score df from complete df."""
    score_df = df["scores"].apply(
        lambda json_str: pd.Series(json.loads(json_str))
        if isinstance(json_str, str)
        else lambda json_str: pd.Series(json_str, dtype=float)
    )
    if filters is not None:
        score_df = score_df.drop(
            columns=[c for c in score_df.columns if any(c.startswith(f) for f in filters)]
        )
        score_df = score_df.drop(
            columns=[c for c in score_df.columns if any(c.endswith(f) for f in filters)]
        )

    return score_df


def get_combo_hash(combo):
    """Convert combo values to hash for saving traces."""
    return sha256(json.dumps(combo).encode()).hexdigest()


def plot_traces(trace_df, trace_path="traces", pdf_filename="traces.pdf"):
    """Plot traces from df, with highlights on rows with trace_highlight = True.

    Args:
        trace_df (DataFrame): contains list of combos with traces to plot
        trace_path (str): path to folder with traces in .pkl
        pdf_filename (str): name of pdf to save
    """
    COLORS = cycle(["r"] + [f"C{i}" for i in range(10)])
    trace_df = trace_df.copy()  # prevents annoying panda warnings
    if "trace_highlight" not in trace_df.columns:
        trace_df["trace_highlight"] = True
    for index in trace_df.index:
        if trace_df.loc[index, "trace_highlight"]:
            c = next(COLORS)

        if "trace_data" in trace_df.columns:
            trace_path = trace_df.loc[index, "trace_data"]
        else:
            combo_hash = get_combo_hash(trace_df.loc[index])
            trace_path = Path(trace_path) / ("trace_id_" + str(combo_hash) + ".pkl")

        with open(trace_path, "rb") as f:
            trace = pickle.load(f)
            if isinstance(trace, list):
                trace = trace[1]  # newer version the response are here
            for protocol, response in trace.items():
                if isinstance(response, TimeVoltageResponse):
                    if trace_df.loc[index, "trace_highlight"]:
                        label = trace_df.loc[index, "name"]
                        lw = 1
                        zorder = 1
                    else:
                        label = None
                        c = "0.5"
                        lw = 0.5
                        zorder = -1

                    plt.figure(protocol, figsize=(15, 7))
                    plt.plot(
                        response["time"],
                        response["voltage"],
                        label=label,
                        c=c,
                        lw=lw,
                        zorder=zorder,
                    )

    with PdfPages(pdf_filename) as pdf:
        for fig_id in plt.get_fignums():
            fig = plt.figure(fig_id)
            plt.legend(loc="best")
            plt.suptitle(fig.get_label())
            pdf.savefig()
            plt.close()
