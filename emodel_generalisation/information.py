"""Module to compute information theory on mcmc sampling of emodels."""

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


import itertools
import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel
from joblib import delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from tqdm import tqdm

from emodel_generalisation import FEATURE_LABELS_LONG
from emodel_generalisation import PARAM_LABELS
from emodel_generalisation.utils import cluster_matrix

try:
    import jpype as jp
except ModuleNotFoundError:
    pass


# pragma: no cover
logger = logging.getLogger(__name__)


def setup_jidt(jarlocation="/gpfs/bbp.cscs.ch/home/arnaudon/code/jidt/infodynamics.jar"):
    """Setup the java env for jidt code."""
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarlocation)
        jp.attachThreadToJVM()


def get_jidt_calc(tpe="MI", algo_type="gaussian"):
    """Get the jidt information theory calculator of given type."""
    if tpe == "MI":
        if algo_type == "kraskov":
            _cls = (
                "infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1"
            )
        elif algo_type == "gaussian":
            _cls = "infodynamics.measures.continuous.gaussian.MultiInfoCalculatorGaussian"
        else:
            raise Exception(f"algo_type {algo_type} not understood")

    elif tpe == "Oinfo":
        if algo_type == "kraskov":
            _cls = "infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov"
        elif algo_type == "gaussian":
            _cls = "infodynamics.measures.continuous.gaussian.OInfoCalculatorGaussian"
        else:
            raise Exception(f"algo_type {algo_type} not understood")
    else:
        raise Exception(f"type {type} not understood")

    _id = _cls.rfind(".")
    _id1 = _id + 1
    return getattr(jp.JPackage(_cls[:_id]), _cls[_id1:])()


def _get_jidt_info_calc(correlation_type, order=2):
    """Create  pair correlation function of given type."""
    calc = get_jidt_calc(tpe=correlation_type)

    def info_calc(x):
        calc.initialise(order)
        calc.setObservations(x)
        calc.setProperty("NUM_THREADS", str(1))  # for kraskov
        return calc.computeAverageLocalOfObservations()

    return info_calc


def log(x, unit="nats"):
    """Log function."""
    if unit == "nats":
        return np.log(x)
    if unit == "bits":
        return np.log2(x)
    raise Exception("Unknown unit")


def oinfo_gaussian(x):
    """Compute Oinfo with guaussian approximation."""
    cov = np.cov(x.T)
    dim = len(cov)
    oinfo = (dim - 2) * log(np.linalg.det(cov))
    for i in range(dim):
        idx = list(range(dim))
        del idx[i]
        oinfo += log(cov[i][i]) - log(np.linalg.det(cov[:, idx][idx, :]))
    return 0.5 * oinfo


def mi_gaussian(x):
    """MI with gaussian approximation."""
    cov = np.cov(x.T)
    mi = -log(np.linalg.det(cov)) + log(cov[0][0]) + log(cov[1][1])
    return 0.5 * mi


def rsi_gaussian(x):
    """RSI calculation with gaussians (assuming first element is y)."""
    cov = np.cov(x.T)
    cov_X = cov[1:][:, 1:]
    dim = len(cov)

    rsi = (dim - 2) * log(cov[0, 0])
    rsi += sum(log(np.diag(cov_X))) - log(np.linalg.det(cov_X))
    for i in range(dim - 1):
        rsi -= log(np.linalg.det(cov[[0, i + 1]][:, [0, i + 1]]))
    rsi += log(np.linalg.det(cov))
    return 0.5 * rsi


def _get_info_calc(correlation_type, order=2):
    """Create  pair correlation function of given type."""
    if order == 2:
        return mi_gaussian
    if correlation_type == "Oinfo":
        return oinfo_gaussian
    if correlation_type == "RSI":
        return rsi_gaussian
    raise Exception("Unknown correlation_type")


def _compute_higher_order_single_feature(
    param_tuples,
    df_1,
    df_2,
    order,
    correlation_type="MI",
    with_params=False,
    with_jidt=False,
):
    """Internal computation of higher order correlations for single feature."""
    if with_jidt:
        setup_jidt()
        info_calc = _get_jidt_info_calc(correlation_type, order=order)
    else:
        info_calc = _get_info_calc(correlation_type, order=order)

    if with_params:
        first_col_name = "param_0"
        col_shift = 1
    else:
        first_col_name = "feature"
        col_shift = 0

    gid = 0
    corr_df = pd.DataFrame()
    for first_col, param_tuple in param_tuples:
        try:
            _first_col = df_1.columns[first_col]
        except Exception as exc:
            raise Exception(first_col, df_1) from exc
        col_1 = df_1.columns[first_col]
        col_2 = df_2.columns[tuple([param_tuple])]
        x1 = df_1[col_1].to_numpy()
        if col_1 == "features":
            x1 = (x1 - x1.mean()) / x1.std()
        x = np.hstack([x1[:, np.newaxis], df_2[col_2].to_numpy()])
        corr = info_calc(x)

        corr_df.loc[gid, first_col_name] = _first_col
        for i, _col in enumerate(col_2):
            corr_df.loc[gid, f"param_{col_shift + i}"] = _col
        corr_df.loc[gid, f"{correlation_type}"] = corr
        gid += 1

    return corr_df


def compute_higher_order(
    df,
    order=3,
    column_1="features",
    column_2="normalized_parameters",
    correlation_type="MI",
    n_workers=50,
    batch_size=100,
    param_tuples=None,
):
    """Compute higher order IT."""
    _df_1 = df[column_1]
    _df_1 = _df_1.T[_df_1.std().T > 0].T
    _df_2 = df[column_2 or column_1]

    if param_tuples is None:
        if column_1 != column_2:
            _param_tuples = itertools.combinations(range(len(_df_2.columns)), order - 1)
            param_tuples = list(itertools.product(list(range(len(_df_1.columns))), _param_tuples))
        else:
            _param_tuples = itertools.combinations(range(len(_df_2.columns)), order)
            param_tuples = [[p[0], p[1:]] for p in _param_tuples]

    print(f"{len(param_tuples)} tuples to compute")
    param_tuples = np.array_split(
        np.array(param_tuples, dtype="object"),
        len(param_tuples) // min(batch_size, len(param_tuples)),
    )

    _compute = partial(
        _compute_higher_order_single_feature,
        df_1=_df_1,
        df_2=_df_2,
        order=order,
        correlation_type=correlation_type,
        with_params=column_1 == "normalized_parameters",
    )
    return pd.concat(
        list(
            Parallel(n_workers, verbose=10)(delayed(_compute)(_tuples) for _tuples in param_tuples)
        )
    )


def compute_higher_orders(
    df,
    min_order=3,
    max_order=5,
    column_1="features",
    column_2="normalized_parameters",
    correlation_type="MI",
    n_workers=50,
    batch_size=100,
    top=100,
    min_order_select=3,
    output_folder="IT_data",
    with_largests=True,
):
    """Compute higher order IT measures.

    Args:
        df (dataframe): MCMC output dataframe
        split (float): max cost to filter dataframe
        max_order (int): max order to compute
        column_1 (str): name of column one (features for example)
        columns_2 (str): usually param column
        correlation_type (str): MI/Oinfo
        n_workers (int): number of parallel workers to use (to many leads to memory error)
        batch_size (int): number of IT evaluation for each workers
        min_order_select (int): after which order we start to only use top/botom best tuples
        output_folder (str): folder to save .csv for each order
    """
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    for order in range(min_order, max_order + 1):
        print(f"Computing order {order}")
        param_tuples = None
        if order > min_order_select:
            param_tuples = create_reduced_tuple_set(
                df,
                output_folder,
                order,
                column_1=column_1,
                column_2=column_2,
                corr_type=correlation_type,
                top=top,
                with_largests=with_largests,
            )

        compute_higher_order(
            df,
            order,
            column_1=column_1,
            column_2=column_2,
            correlation_type=correlation_type,
            n_workers=n_workers,
            batch_size=batch_size,
            param_tuples=param_tuples,
        ).to_csv(
            f"{output_folder}/col1_{column_1}_{correlation_type}_order_{order}.csv", index=False
        )


def create_reduced_tuple_set(
    df,
    data_folder,
    order,
    column_1="features",
    column_2="normalized_parameters",
    corr_type="Oinfo",
    top=100,
    with_largests=True,
):
    """Create a reduced tuple set."""
    if column_1 == "features":
        return create_reduced_tuple_set_features(
            df, data_folder, order, column_1, column_2, corr_type, top
        )
    return create_reduced_tuple_set_parameters(
        df, data_folder, order, column_1, column_2, corr_type, top, with_largests
    )


def create_reduced_tuple_set_parameters(
    df,
    data_folder,
    order,
    column_1="features",
    column_2="normalized_parameters",
    corr_type="Oinfo",
    top=100,
    with_largests=True,
):
    """Select reduced set of tuples using top previous tuples."""
    data = pd.read_csv(f"{data_folder}/col1_{column_1}_{corr_type}_order_{order-1}.csv")
    top_param_tuples = (
        data.sort_values(by=corr_type)[:top].drop(columns=[corr_type]).to_numpy().tolist()
    )
    if with_largests:
        top_param_tuples += (
            data.sort_values(by=corr_type)[-top:].drop(columns=[corr_type]).to_numpy().tolist()
        )
    params = df[column_2].columns
    param_tuples = set(
        tuple(sorted([p[0]] + p[1]))
        for p in itertools.product(params, top_param_tuples)
        if p[0] not in p[1]
    )
    params_conv = {p: i for i, p in enumerate(params)}
    return [[params_conv[p[0]], [params_conv[_p] for _p in p[1:]]] for p in param_tuples]


def create_reduced_tuple_set_features(
    df,
    data_folder,
    order,
    column_1="features",
    column_2="normalized_parameters",
    corr_type="Oinfo",
    top=100,
    with_largests=True,
):
    """Select reduced set of tuples using lower percentile of previous order."""
    data = pd.read_csv(f"{data_folder}/col1_{column_1}_{corr_type}_order_{order-1}.csv")

    params = df[column_2].columns
    params_conv = {p: i for i, p in enumerate(params)}
    features = df["features"].columns
    tuples = []
    for feat_id, feat in enumerate(features):
        _data = data[data["feature"] == feat].drop(columns=["feature"])

        top_param_tuples = (
            _data.sort_values(by=corr_type)[:top].drop(columns=[corr_type]).to_numpy().tolist()
        )
        if with_largests:
            top_param_tuples += (
                _data.sort_values(by=corr_type)[-top:].drop(columns=[corr_type]).to_numpy().tolist()
            )
        param_tuples = set(
            tuple(sorted([p[0]] + p[1]))
            for p in itertools.product(params, top_param_tuples)
            if p[0] not in p[1]
        )
        tuples += [[feat_id, [params_conv[_p] for _p in p]] for p in param_tuples]
    return tuples


def plot_tuple_distributions(
    data_folder="data",
    figure_name="IT_corr.pdf",
    correlation_type="Oinfo",
    min_order=3,
    max_order=20,
    column_1="features",
    with_min=True,
    with_max=True,
    n_top_tuples=100,
    tuple_freq_thresh=0.01,
):
    """Plot tuple distributions."""
    data = {}
    for order in range(min_order, max_order + 1):
        try:
            data[order] = pd.read_csv(
                f"{data_folder}/col1_{column_1}_{correlation_type}_order_{order}.csv"
            )
        except Exception:  # pylint: disable=broad-except
            pass

    def get_param_bars(param_tuple, correlation_type):
        for i, _ in enumerate(param_tuple):
            param_tuple[i]["order"] = i
        param_tuple = pd.concat(param_tuple)
        if column_1 == "features":
            param_tuple = param_tuple.drop(columns=["feature", correlation_type])
        param_tuple = param_tuple.reset_index(drop=True)
        unique_params = np.unique(
            [t for t in param_tuple.to_numpy().flatten() if isinstance(t, str)]
        )
        param_bars = pd.DataFrame(
            index=param_tuple.order.unique(), columns=unique_params, dtype=float
        )
        param_bars = param_bars.fillna(0)
        for order in param_bars.index:
            for _param in param_tuple[param_tuple.order == order].to_numpy():
                for param in _param:
                    if isinstance(param, str):
                        param_bars.loc[order, param] += 1.0
        param_bars = param_bars.T
        param_bars["n"] = param_bars.sum(1).to_numpy()
        param_bars = param_bars.sort_values(by="n", ascending=False).drop(columns=["n"])
        param_bars.columns = np.array(param_bars.columns) + min(all_orders)
        param_bars[param_bars < tuple_freq_thresh * param_bars.max(0)] = np.nan
        param_bars.index = [PARAM_LABELS.get(p, p) for p in param_bars.index]
        return param_bars.dropna(axis=0, how="all")

    def plot(data, col=None, col_val=None):
        max_data = []
        min_data = []
        strip_df = []
        max_tuple = []
        min_tuple = []
        min_top_params = {}
        max_top_params = {}
        for order in all_orders:
            df = data[order]
            if col is not None:
                df = df[df[col] == col_val]
            _strip_df = pd.DataFrame()
            _strip_df["data"] = df[correlation_type].to_list()
            _strip_df["order"] = order
            strip_df.append(_strip_df.sort_values(by="data"))
            if with_max:
                strip_df.append(_strip_df.sort_values(by="data"))
            min_data.append(df[correlation_type].min())
            drop_cols = ["features", correlation_type] if col == "features" else [correlation_type]
            max_top_params[order] = (
                df.sort_values(by=correlation_type).tail(1).drop(columns=drop_cols).to_numpy()[0]
            )
            max_top_params[order] = [PARAM_LABELS.get(p, p) for p in max_top_params[order]]
            min_top_params[order] = (
                df.sort_values(by=correlation_type).head(1).drop(columns=drop_cols).to_numpy()[0]
            )
            min_top_params[order] = [PARAM_LABELS.get(p, p) for p in min_top_params[order]]
            if with_min:
                min_tuple.append(df.loc[df[correlation_type].sort_values()[:n_top_tuples].index])
            max_data.append(df[correlation_type].max())
            if with_max:
                max_tuple.append(df.loc[df[correlation_type].sort_values()[-n_top_tuples:].index])

        if with_min:
            min_param_bars = get_param_bars(min_tuple, correlation_type)
            min_param_bars.index.name = "param"

        if with_max:
            max_param_bars = get_param_bars(max_tuple, correlation_type)
            max_param_bars.index.name = "param"

        strip_df = pd.concat(strip_df).reset_index(drop=True)
        if with_max and with_min:
            f, (ax1, ax2, ax3) = plt.subplots(
                3, 1, gridspec_kw={"height_ratios": [2.0, 2.5, 2.0]}, figsize=(10, 20)
            )
        elif with_min:
            f, (ax2, ax3) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0}, figsize=(10, 15)
            )
        else:
            f, (ax2) = plt.subplots(1, 1, gridspec_kw={"height_ratios": [1.0], "hspace": 0})

        ax2.fill_between(np.arange(len(min_data)), min_data, max_data, color="k", alpha=0.2)
        ax2.axhline(0, c="k", ls="--")
        for i in range(len(min_data)):
            ax2.axvline(i, c="k", lw=0.5)
        sns.stripplot(data=strip_df, ax=ax2, x="order", y="data", size=3, color="k")

        ax2.set_ylabel(f"{correlation_type}")
        ax2.set_xlabel("")
        if col_val is not None:
            plt.suptitle(col_val)
        if with_max:
            sns.heatmap(max_param_bars, ax=ax1, cmap="Blues", cbar=False, yticklabels=True)

            for order in all_orders:
                idx = max_param_bars.reset_index().index[
                    max_param_bars.index.isin(max_top_params[order])
                ]
                ax1.scatter(len(idx) * [order - all_orders[0] + 0.5], idx + 0.5, c="r")
            ax1.set_ylabel("parameter frequency")
        if with_min:
            min_param_bars = min_param_bars.dropna(axis=0, how="all")
            sns.heatmap(min_param_bars, ax=ax3, cmap="Blues", cbar=False, yticklabels=True)

            for order in all_orders:
                idx = min_param_bars.reset_index().index[
                    min_param_bars.index.isin(min_top_params[order])
                ]
                ax3.scatter(len(idx) * [order - all_orders[0] + 0.5], idx + 0.5, c="r")

            ax3.set_ylabel("parameter frequency")
            ax3.set_xlabel("order")

        return f

    all_orders = list(data.keys())
    if "feature" in data[all_orders[0]].columns:
        with PdfPages(f"{figure_name}_{correlation_type}.pdf") as pdf:
            for feature in tqdm(data[all_orders[0]].feature.unique()[::-1]):
                f = plot(data, "feature", feature)
                pdf.savefig(f, bbox_inches="tight")
                plt.close("all")
    else:
        plot(data)
        plt.savefig(f"{figure_name}_{correlation_type}.pdf", bbox_inches="tight")
        plt.savefig(f"{figure_name}_{correlation_type}.png", bbox_inches="tight")


# older functions, to deprecate


def _get_pair_correlation_function(correlation_type):
    """Create  pair correlation function of given type."""
    if correlation_type == "pearson":

        def corr_f(x, y):
            return pearsonr(x, y)[0]

    elif correlation_type == "MI":
        calc_mi = _get_info_calc("MI")

        def corr_f(x, y):
            return calc_mi(np.array([x, y]).T)

    else:
        raise Exception("Correlation type not understood")
    return corr_f


def reduce_matrix_percentile(df, percentile, data=None):
    """Reduce matrix percentile."""
    if data is None:
        data = abs(df.to_numpy().flatten())
    df[abs(df) < np.percentile(data, percentile)] = np.nan
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df[df.isna()] = 0
    return df


def plot_pair_correlations(
    df,
    split=None,
    min_corr=0.3,
    column_1="normalized_parameters",
    column_2=None,
    filename="parameter_pairs.pdf",
    clip=0.4,
    correlation_type="pearson",
    with_plots=False,
    plot_top_only_perc=None,
):
    """Scatter plots of pairs with pearson larger than min_corr, and pearson correlation matrix.

    If column_2 is provided, the correlation will be non-square and no clustering will be applied.
    Args:
        min_corr (float): minimum correlation for plotting scatter plot
        clip (float): value to clip correlation matrix
    """
    if correlation_type != "pearson":
        try:
            setup_jidt()
        except Exception:  # pylint: disable=broad-except,bare-except
            pass
    corr_f = _get_pair_correlation_function(correlation_type)
    signed = correlation_type == "pearson"

    if split is not None:
        _df = df[df["cost"] < split]
    else:
        _df = df

    _df_1 = _df[column_1]
    _df_2 = _df[column_2 or column_1]

    if column_1 == column_2:
        pairs = list(itertools.combinations(_df_1.columns, 2))
    else:
        pairs = list(itertools.product(_df_1.columns, _df_2.columns))

    if with_plots:
        pdf = PdfPages(filename)

    corr_df = pd.DataFrame()
    corrs = []
    for col_1, col_2 in tqdm(pairs):
        x = _df_1[col_1].to_numpy()
        y = _df_2[col_2].to_numpy()
        x[np.isnan(x)] = 0
        y[np.isnan(y)] = 0
        corr = corr_f(x, y)
        corr_df.loc[col_1, col_2] = corr
        corrs.append(corr)
        if column_1 == column_2:
            corr_df.loc[col_2, col_1] = corr
            corr_df.loc[col_1, col_1] = 0
            corr_df.loc[col_2, col_2] = 0
        if with_plots and abs(corr) > min_corr:
            plt.figure(figsize=(5, 4))
            cost = None
            plt.scatter(x, y, marker=".", c=cost, s=0.5)
            plt.colorbar()
            plt.scatter(x[0], y[0], marker="o", c="r", s=0.5)
            plt.xlabel(col_1)
            plt.ylabel(col_2)
            plt.gca().set_rasterized(True)
            plt.suptitle(f"{correlation_type} = {np.around(corr, 2)}")
            pdf.savefig(bbox_inches="tight", dpi=200)
            plt.close()

    if with_plots:
        pdf.close()

    if plot_top_only_perc is not None:
        corr_df = reduce_matrix_percentile(corr_df, plot_top_only_perc, data=corrs)

    corr_df = corr_df.sort_index(axis=0).sort_index(axis=1)
    if column_1 == column_2:
        corr_df[corr_df.isna()] = 0.001
        sorted_labels = cluster_matrix(corr_df.abs())
        corr_df = corr_df.loc[sorted_labels, sorted_labels]

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    corr_df[abs(corr_df) < clip] = np.nan
    if signed:
        sns.heatmap(
            data=corr_df,
            ax=ax,
            # vmin=-1, vmax=1,
            cmap="bwr",
            linewidths=1,
            linecolor="k",
            xticklabels=True,
            yticklabels=True,
        )
    else:
        corr_df[corr_df > 5] = 5
        sns.heatmap(
            data=np.log10(0.1 + corr_df),
            ax=ax,
            cmap="Blues",
            linewidths=1,
            linecolor="k",
            cbar_kws={"label": "log10(MI)"},
            xticklabels=True,
            yticklabels=True,
        )
        if column_1 == "features":
            ticks = [FEATURE_LABELS_LONG[p] for p in corr_df.index]
        else:
            ticks = [PARAM_LABELS[p] for p in corr_df.index]
        ax.set_yticklabels(ticks)
        if column_2 == "features":
            ticks = [FEATURE_LABELS_LONG[p] for p in corr_df.columns]
        else:
            ticks = [PARAM_LABELS[p] for p in corr_df.columns]
        ax.set_xticklabels(ticks)
    plt.savefig(
        Path(str(Path(filename).with_suffix("")) + "_matrix").with_suffix(Path(filename).suffix),
        bbox_inches="tight",
    )
    return corr_df


def reduce_features(df, threshold=0.9):
    """Reduce number of feature to non-correlated features."""
    selected_features = []
    feature_map = defaultdict(list)
    for feature1 in sorted(df.index):
        to_add = True
        for feature2 in selected_features:
            if df.loc[feature1, feature2] > threshold:
                feature_map[feature2].append(feature1)
                to_add = False
        if to_add:
            selected_features.append(feature1)
    print(f"Found {len(selected_features)} out of {len(df.index)}")
    return selected_features, feature_map
