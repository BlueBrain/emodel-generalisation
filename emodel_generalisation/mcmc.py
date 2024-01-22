"""Module to sample electrical model parameter space with MCMC."""

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
import json
import sys
import traceback
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits import axisartist
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from emodel_generalisation import ALL_LABELS
from emodel_generalisation import PARAM_LABELS
from emodel_generalisation.information import mi_gaussian
from emodel_generalisation.information import rsi_gaussian
from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.model.evaluation import get_evaluator_from_access_point
from emodel_generalisation.parallel import evaluate
from emodel_generalisation.parallel import init_parallel_factory
from emodel_generalisation.utils import cluster_matrix

# pylint: disable=too-many-lines,too-many-locals

matplotlib.use("Agg")


class MarkovChain:
    """Class to setup and run a markov chain on emodel parameter space."""

    def __init__(
        self,
        n_steps=100,
        result_df_path="result.csv",
        temperature=1.0,
        proposal_params=None,
        emodel=None,
        access_point=None,
        stochasticity=False,
        mcmc_type="metropolis_hastings",
        weights=None,
        seed=42,
        frozen_params=None,
        random_initial_parameters=True,
        mcmc_log_file="mcmc_log.txt",
        cost_type="max",
        resume=False,
    ):
        """Initialise the markov chain object."""
        self.result_df_path = result_df_path
        self.n_steps = n_steps
        self.temperature = temperature
        self.mcmc_type = mcmc_type
        self.weights = weights if weights is not None else {}
        self.stochasticity = stochasticity
        self.frozen_params = frozen_params
        self.mcmc_log_file = mcmc_log_file
        self.cost_type = cost_type
        self.resume = resume

        self.seed = seed
        np.random.seed(self.seed)
        self.float_precision = 12

        if proposal_params is None:
            proposal_params = {"type": "normal", "std": 0.02}
        self.proposal_params = proposal_params
        self.emodel = emodel
        self.access_point = access_point

        self._evaluator = None
        evaluator = self.get_evaluator()
        self.param_names = [param.name for param in evaluator.params if not param.frozen]

        self.lbounds = {
            param.name: param.lower_bound for param in evaluator.params if not param.frozen
        }
        self.ubounds = {
            param.name: param.upper_bound for param in evaluator.params if not param.frozen
        }
        self.bounds = {
            "center": {p: 0.5 * (self.ubounds[p] + self.lbounds[p]) for p in self.param_names},
            "width": {p: 0.5 * (self.ubounds[p] - self.lbounds[p]) for p in self.param_names},
        }
        self.random_initial_parameters = random_initial_parameters
        if self.random_initial_parameters:
            self.initial_parameters = self.get_random_parameters()
        else:
            if "params" in access_point.final[self.emodel]:
                self.initial_parameters = self.access_point.final[self.emodel]["params"]
            else:
                self.initial_parameters = self.access_point.final[self.emodel]["parameters"]

        self.initial_parameters = {
            p: v for p, v in self.initial_parameters.items() if p in self.param_names
        }

        self.feature_names = [obj.name for obj in evaluator.fitness_calculator.objectives]

        _cols = [("parameters", param) for param in self.param_names]
        _cols += [("normalized_parameters", param) for param in self.param_names]
        _cols += [("features", feat) for feat in self.feature_names]
        _cols += [("scores", feat) for feat in self.feature_names]
        if self.resume:
            self.result_df = pd.read_csv(self.result_df_path, header=[0, 1])
        else:
            self.result_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(_cols))
            # create single columns
            self.result_df["cost"] = None
            self.result_df["accept_rate"] = None
            self.result_df["probability"] = None
            self.result_df["a"] = None
            self.result_df["ratio"] = None

            self.result_df.to_csv(self.result_df_path, index=None)

        self.csv_file = open(self.result_df_path, "a")  # pylint: disable=consider-using-with
        self.mcmc_log = open(self.mcmc_log_file, "a")  # pylint: disable=consider-using-with
        self.accepted = True

        # we build multi normal such as std is expected radius of proposal
        _mn = multivariate_normal(mean=len(self.param_names) * [0], cov=1)
        std = np.mean(np.linalg.norm(_mn.rvs(size=1000), axis=1))
        self.multi_normal = multivariate_normal(
            mean=len(self.param_names) * [0], cov=self.proposal_params["std"] / std
        )

    def get_random_parameters(self):
        """Get random parameter to initialise a chain."""
        normed_params = {p: np.random.uniform(-1.0, 1.0) for p in self.param_names}
        return self._un_normalize_parameters(normed_params)

    def get_evaluator(self):
        """Get the evaluator.

        We do this so that we don't have to unfreeze params if the evaluation crashes neuron.
        """
        self.access_point.settings = self.access_point.get_settings(self.emodel)

        evaluator = get_evaluator_from_access_point(
            self.emodel,
            self.access_point,
            stochasticity=self.stochasticity,
            timeout=1e10,
        )

        # unfreeze all to be sure
        for i, _param in enumerate(evaluator.params):
            evaluator.params[i].unfreeze()

        # freeze params if any in frozen_params
        if self.frozen_params is not None:
            for f_param in self.frozen_params:
                for i, _param in enumerate(evaluator.params):
                    if _param.name == f_param:
                        evaluator.params[i].freeze(self.frozen_params[f_param])

        return evaluator

    def _un_normalize_parameters(self, parameters):
        return {
            p: v * self.bounds["width"][p] + self.bounds["center"][p] for p, v in parameters.items()
        }

    def _normalize_parameters(self, parameters):
        return {
            p: (v - self.bounds["center"][p]) / self.bounds["width"][p]
            for p, v in parameters.items()
        }

    def _probability_distribution(self, cost):
        """Convert cost function to Boltzman distribution with given temperature."""
        return np.exp(-cost / self.temperature)

    def _propose_parameters(self, parameters):
        """Propose new parameters."""
        normalized_parameters = self._normalize_parameters(parameters)
        shift = self.multi_normal.rvs(size=1)
        proposed = {p: v + shift[i] for i, (p, v) in enumerate(normalized_parameters.items())}
        return self._un_normalize_parameters(proposed)

    def _round_parameters(self, parameters):
        """Round parameters to ensure reproducibility."""
        return {p: np.around(v, self.float_precision) for p, v in parameters.items()}

    def _evaluate(self, parameters):
        """Run evaluation."""
        evaluator = self.get_evaluator()
        parameters = self._round_parameters(parameters)
        responses = evaluator.run_protocols(evaluator.fitness_protocols.values(), parameters)
        scores = evaluator.fitness_calculator.calculate_scores(responses)
        values = evaluator.fitness_calculator.calculate_values(responses)

        for f, val in values.items():
            if isinstance(val, np.ndarray) and len(val) > 0:
                try:
                    values[f] = np.nanmean(val)
                except (AttributeError, TypeError):
                    values[f] = None
            else:
                values[f] = None

        cost = eval(self.cost_type)(self.weights.get(f, 1.0) * scores[f] for f in scores)
        probability = self._probability_distribution(cost)
        return {
            "parameters": parameters,
            "cost": cost,
            "probability": probability,
            "scores": scores,
            "values": values,
        }

    def _metropolis_hastings_step(self, current, depth=0):
        """Run a single metropolis-hastings step."""
        _proposed = self._propose_parameters(current["parameters"])

        # prevent bias if we hit boundary
        lower = [_proposed[p] - self.lbounds[p] for p in _proposed]
        upper = [self.ubounds[p] - _proposed[p] for p in _proposed]
        if min(lower) < 1e-9 or min(upper) < 1e-9:
            if depth > 1000:
                if min(lower) < 1e-9:
                    param = list(_proposed.keys())[np.argmin(lower)]
                if min(upper) < 1e-9:
                    param = list(_proposed.keys())[np.argmin(upper)]
                print(
                    f"outside, depth={depth}, param={param}, value={_proposed[param]}",
                    file=self.mcmc_log,
                    flush=True,
                )

                _proposed = {
                    p: np.clip(val, self.lbounds[p], self.ubounds[p])
                    for p, val in _proposed.items()
                }
            else:
                return self._metropolis_hastings_step(current, depth=depth + 1)

        try:
            proposed = self._evaluate(_proposed)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{exc}, depth={depth}", file=self.mcmc_log, flush=True)
            return current

        a = np.random.uniform(0, 1)
        ratio = proposed["probability"] / current["probability"]
        self.accepted = True
        if ratio < a:
            # we reject
            proposed = current
            self.accepted = False

        proposed["a"] = a
        proposed["ratio"] = ratio

        return proposed

    def _run_one_step(self, current):
        """Run one MCMC step of chosen algorithm."""
        if self.mcmc_type == "metropolis_hastings":
            return self._metropolis_hastings_step(current)
        raise Exception("mcmc type not implemented")

    def _append_result(self, step, result, accept_rate):
        """Append result to main dataframe and save."""
        self.result_df.loc[step, "cost"] = result["cost"]
        self.result_df.loc[step, "accept_rate"] = accept_rate
        self.result_df.loc[step, "probability"] = result["probability"]
        self.result_df.loc[step, "a"] = result["a"]
        self.result_df.loc[step, "ratio"] = result["ratio"]

        self.result_df.loc[step, [("parameters", p) for p in result["parameters"]]] = list(
            result["parameters"].values()
        )
        normed_params = self._normalize_parameters(result["parameters"])
        self.result_df.loc[step, [("normalized_parameters", p) for p in normed_params]] = list(
            normed_params.values()
        )
        for f, v in result["values"].items():
            self.result_df.loc[step, [("features", f)]] = v
        for f, s in result["scores"].items():
            self.result_df.loc[step, [("scores", f)]] = s

        # to lower the I/O we just append lines to the .csv
        print(
            ",".join(str(s) if s is not None else "" for s in self.result_df.loc[step].to_list()),
            file=self.csv_file,
            flush=True,
        )
        # self.result_df.to_csv(self.result_df_path, index=False)

    def run(self, depth=0):
        """Run the MCMC."""
        try:
            current = self._evaluate(self.initial_parameters)
            current["a"] = 1.0
            current["ratio"] = 1.0
        except Exception as exc:  # pylint: disable=broad-except
            exception = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"{exception}, depth={depth}", file=self.mcmc_log, flush=True)
            if self.random_initial_parameters and depth < 100:
                self.initial_parameters = self.get_random_parameters()
                return self.run(depth + 1)
            else:
                raise exc

        current["probability"] = self._probability_distribution(current["cost"])

        if self.resume:
            step_0 = self.result_df.index[-1]
            n_accepted = self.result_df["n_accepted"].to_numpy()
            n_accepted = n_accepted[~np.isnan(n_accepted)][-1]
        else:
            step_0 = 0
            n_accepted = 0

        for step in range(step_0, self.n_steps + step_0):
            n_accepted += int(self.accepted)
            accept_rate = np.round(100.0 * (n_accepted / (step + 1)), 1)
            if self.accepted:
                self._append_result(step, current, accept_rate)

            _log = ""
            _log += f"Emodel {self.emodel}, "
            _log += f"seed {self.seed}, step {step}, cost {current['cost']}, "
            _log += f"acceptance rate {accept_rate}"
            print(_log, file=self.mcmc_log, flush=True)

            current = self._run_one_step(current)

        return self.result_df


def _eval_one_chain(row, params=None):
    """Internal function to evaluate one chain for parallel processing."""
    MarkovChain(
        **params,
        emodel=row["emodel"],
        result_df_path=row["result_df_path"],
        seed=row["seed"],
    ).run()
    return {"done": True}


def run_several_chains(
    n_chains=50,
    n_steps=100,
    results_df_path="chains",
    run_df_path="run_df.csv",
    temperature=1.0,
    proposal_params=None,
    access_point=None,
    emodel_dir=None,
    recipes_path=None,
    final_path=None,
    legacy_dir_structure=True,
    emodel=None,
    with_seeds=False,
    stochasticity=False,
    mcmc_type="metropolis_hastings",
    parallel_lib="multiprocessing",
    frozen_params=None,
    random_initial_parameters=True,
    mcmc_log_file="mcmc_log.txt",
    weights=None,
    chain_df=None,
):
    """Main function to call to run several chains in parallel."""
    if isinstance(parallel_lib, str):
        parallel_factory = init_parallel_factory(parallel_lib)
    else:
        parallel_factory = parallel_lib
    Path(results_df_path).mkdir(exist_ok=True, parents=True)
    if access_point is None:
        access_point = AccessPoint(
            emodel_dir=emodel_dir,
            recipes_path=recipes_path,
            final_path=final_path,
            legacy_dir_structure=legacy_dir_structure,
            with_seeds=with_seeds,
        )
    _eval = partial(
        _eval_one_chain,
        params={
            "n_steps": n_steps,
            "temperature": temperature,
            "proposal_params": proposal_params,
            "stochasticity": stochasticity,
            "mcmc_type": mcmc_type,
            "frozen_params": frozen_params,
            "random_initial_parameters": random_initial_parameters,
            "access_point": access_point,
            "mcmc_log_file": mcmc_log_file,
            "weights": weights,
            "resume": chain_df is not None,
        },
    )
    if chain_df is None:
        _dfs = []
        emodels = access_point.final.keys() if emodel is None else [emodel]
        for i, _emodel in enumerate(emodels):
            _df = pd.DataFrame()
            _df["chain_id"] = list(range(i, i + n_chains))
            _df["seed"] = list(range(i, i + n_chains))
            txt = f"{results_df_path}/results_df_emodel_{_emodel}"
            _df["result_df_path"] = _df["chain_id"].apply(
                lambda _id, _txt=txt: f"{_txt}_id_{str(_id)}.csv"
            )
            _df["emodel"] = _emodel
            _dfs.append(_df)
        df = pd.concat(_dfs).reset_index(drop=True)
        df.to_csv(run_df_path, index=False)
    else:
        df = pd.read_csv(chain_df)
    df = evaluate(df, _eval, parallel_factory=parallel_factory, new_columns=[["done", False]])

    # save again to get the exceptions if any
    df.to_csv(run_df_path, index=False)


def load_chains(run_df, base_path=".", with_single_origin=False, n_chains=None):
    """Load chains from main run_df file where the first row contains initial condition.

    If run_df is a path to .csv, we will load it.
    """
    if isinstance(run_df, (Path, str)):
        run_df = pd.read_csv(run_df)
    if n_chains is not None:
        run_df = run_df.sample(n_chains)

    # load all but first point
    _dfs = []
    for gid in tqdm(run_df.index):
        res = run_df.loc[gid, "result_df_path"]
        try:
            if with_single_origin:
                _df = pd.read_csv(Path(base_path) / res, header=[0, 1]).loc[1:]
            else:
                _df = pd.read_csv(Path(base_path) / res, header=[0, 1])
            _df["chain_id"] = int(Path(res).stem.split("_")[-1])
            _df["emodel"] = run_df.loc[gid, "emodel"]
            _dfs.append(_df)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Cannot read {res}, {exc}")
    df = pd.concat(_dfs)
    df = df.rename(columns=lambda name: "" if name.startswith("Unnamed:") else name)
    df = df.reset_index(drop=True)

    if with_single_origin:
        # add the original point in front
        df_orig = pd.read_csv(Path(base_path) / run_df.result_df_path[0], header=[0, 1])
        df_orig = df_orig.rename(columns=lambda name: "" if name.startswith("Unnamed:") else name)
        df_orig["chain_id"] = int(-1)
        df.loc[-1] = df_orig.loc[0]
    return df.sort_index().reset_index(drop=True)


def _plot_distributions(df, split, filename="parameters.pdf", column="normalized_parameters"):
    """Plot distributions below and above a split."""

    _df = df.copy()
    _df.loc[_df["cost"] < split, "cost"] = 0
    _df.loc[_df["cost"] >= split, "cost"] = 1

    plot_df = _df[column].reset_index(drop=True)

    df_1 = plot_df[_df["cost"] == 0].melt()
    df_1["cost"] = f"< {split}"
    df_2 = plot_df[_df["cost"] == 1].melt()
    df_2["cost"] = f">= {split}"
    plot_df = pd.concat([df_1, df_2])
    if column == "scores":
        plot_df["value"] = np.clip(plot_df["value"], 0, 10)

    n_rows = len(plot_df.variable.unique())
    fig = plt.figure(figsize=(3, int(0.3 * n_rows)))
    ax = plt.gca()
    plot_df["variable"] = plot_df["variable"].apply(lambda p: PARAM_LABELS.get(p, p))
    labels = sorted(plot_df.variable.unique())
    _sns = sns.violinplot(
        data=plot_df,
        bw=0.05,
        orient="h",
        scale="area",
        ax=ax,
        y="variable",
        x="value",
        linewidth=0.1,
        hue="cost",
        split=True,
        order=labels,
        ha="left",
    )

    # set the ticks aligned to the left
    _sns.set_yticklabels(labels, ha="left")
    pad = max(
        ytick.get_window_extent(renderer=fig.canvas.get_renderer()).width
        for ytick in ax.get_yticklabels()
    )
    ax.get_yaxis().set_tick_params(pad=pad - 25)

    for i in range(n_rows):
        plt.axhline(i, c="k", ls="--", lw=0.8)

    if column == "normalized_parameters":
        # plt.axvline(-1, c="k")
        # plt.axvline(1, c="k")
        ax.set_xlim(-1.05, 1.05)
    if column == "scores":
        # plt.axvline(0, c="k")
        # plt.axvline(5, c="k")
        ax.set_xlim(-0.05, 11.00)

    plt.savefig(filename, bbox_inches="tight")


def plot_parameter_distributions(df, split, filename="figures/parameter_distributions.pdf"):
    """Plot parameter distributions below and above a split."""
    _plot_distributions(df, split, filename=filename, column="normalized_parameters")


def plot_score_distributions(df, split, filename="figures/score_distributions.pdf"):
    """Plot score below and above a split."""
    _plot_distributions(df, split, filename=filename, column="scores")


def plot_cost(df, split, filename="figures/costs.pdf"):
    """Plot histogram of costs."""
    plt.figure(figsize=(5, 3))
    df["cost"].plot.hist(bins=200, log=True)
    plt.gca().set_xlim(0, df["cost"].max())

    plt.axvline(split, c="r", label="split")

    plt.legend(loc="best")
    plt.xlabel("cost")
    plt.title(f"total evaluation: {len(df.index)}, below threshold: {len(df[df['cost']<split])}")

    plt.savefig(filename, bbox_inches="tight")


def filter_features(df):
    """Filter redundant features to reduce number of IT computations."""
    features = df["features"].columns.tolist()
    features_to_remove = []
    for feature in features:
        if feature.startswith("Step") and feature.split(".")[0] != "Step_200":
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.startswith("SpikeRec"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.startswith("IV"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.startswith("APWaveform"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.endswith("Spikecount"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.endswith("APlast_amp"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.endswith("third_ISI"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.endswith("fourth_ISI"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))
        if feature.endswith("fifth_ISI"):
            features_to_remove.append(("features", feature))
            features_to_remove.append(("scores", feature))

    return df.drop(columns=features_to_remove)


def filter_local_min(df, dist_split=3, emodel_index=0):
    """Filter emodel by distance to a given emodel, for local minima filtering."""
    if dist_split is None:
        return df
    p = df["normalized_parameters"].to_numpy()
    dists = distance_matrix(p[emodel_index][np.newaxis], p)[0]
    return df[dists < dist_split]


# below some functions to select emodels from mcmc with large distances between them


def select_emodels(df, threshold=4.0, method="local_min"):
    """Select emodels far away from each others in parameter space.

    if method == local_min: we look for local minimum with radius = threshold
    if method == distance: we naively search for points at minimum threshold radius
    """
    p = df["normalized_parameters"].to_numpy()
    dists = distance_matrix(p, p)
    print(f"max dist {np.max(dists)}")

    if method == "local_min":
        emodel_ids = []
        for i, dist in enumerate(dists):
            if (
                np.mean(
                    df.loc[dist < threshold, "cost"].to_numpy() >= df.loc[i, "cost"].to_numpy()[0]
                )
                >= 1.0
            ):
                emodel_ids.append(i)

    if method == "distance":
        emodel_ids = [0]
        for i in range(len(dists)):
            if all(dists[i, j] > threshold for j in emodel_ids):
                emodel_ids.append(i)

    print(f"selected {len(emodel_ids)} emodels")
    return emodel_ids, dists


def plot_selected_emodels(df, emodel_ids, dists, threshold=4):
    """Plot histogram of emodel distances, and clustered distance matrix with selected emodels."""
    plt.figure()
    plt.hist(dists.flatten(), bins=100, log=True)
    plt.axvline(np.mean(dists), c="k")
    plt.savefig("dist_hist.pdf")

    labels = cluster_matrix(pd.DataFrame(dists), distance=True)

    plt.figure()
    plt.imshow(dists[labels, :][:, labels])
    plt.colorbar()
    o = np.argwhere(labels == 0)[0]
    plt.scatter(o, o, c="r")
    _emodel_ids = [np.argwhere(m == labels)[0][0] for m in emodel_ids]
    plt.scatter(_emodel_ids, _emodel_ids, c="m", marker="+")
    plt.savefig("dists.pdf")

    plt.figure()
    with PdfPages("local_min.pdf") as pdf:
        for i, emodel_id in enumerate(emodel_ids):
            close_ids = dists[emodel_id] < threshold
            _costs = df.loc[close_ids, "cost"].to_list()
            _dists = dists[emodel_id, close_ids]
            plt.figure()
            plt.scatter(_dists, _costs, marker=".", color=f"C{i}")
            plt.xlabel("euclidean distance")
            plt.ylabel("cost")
            plt.gca().set_xlim(0, threshold)
            pdf.savefig()
            plt.close()

    plt.figure(figsize=(5, 10))
    y = np.arange(0, len(df["normalized_parameters"].columns))
    for i in emodel_ids:
        if i != 0:
            plt.scatter(df.loc[i, "normalized_parameters"], y, s=30)
    plt.scatter(df.loc[0, "normalized_parameters"], y, marker="+", c="r", s=80)

    ax = plt.gca()
    ax.set_xlim(-1.01, 1.01)
    ax.set_yticks(y)
    ax.set_yticklabels(df["normalized_parameters"].columns)
    plt.savefig("local_params.pdf", bbox_inches="tight")


def save_selected_emodels(df, emodel_ids, emodel="cADpyr_L5TPC", final_path="selected_final.json"):
    """Create a final.json file with selected emodels."""
    final = {}
    for gid in emodel_ids:
        params = df.loc[gid, "parameters"]
        _emodel = df.loc[gid, "emodel"].to_list()[0]
        if _emodel == emodel:
            _emodel = f"{emodel}_{gid}"
        final[_emodel] = {}
        final[_emodel]["params"] = params.to_dict()
        final[_emodel]["score"] = float(df.loc[gid, "cost"].to_list()[0])
        final[_emodel]["fitness"] = df.loc[gid, "scores"].to_dict()
        final[_emodel]["emodel"] = _emodel
    with open(final_path, "w") as f:
        json.dump(final, f, indent=4)


def plot_step_size(run_df, filename="mcmc_stepsize.pdf"):
    """Plot the step size of the first chain in normalized parameter space."""
    _df = pd.read_csv(run_df.loc[0, "result_df_path"], header=[0, 1])
    p = _df["normalized_parameters"].to_numpy()
    plt.figure(figsize=(5, 3))
    diff = np.linalg.norm(np.diff(p, axis=0), axis=1)
    plt.hist(diff, bins=50)
    plt.axvline(np.mean(diff), c="r")
    plt.tight_layout()
    plt.xlabel("mcmc step size")
    plt.savefig(filename)


def plot_full_cost_convergence(df_burnin, df, clip=10, filename="cost_convergence.pdf"):
    """Plot the value of the cost of each chain as a function of iteration."""
    df_burnin["cost"] = np.clip(df_burnin["cost"], 0, clip)
    df["cost"] = np.clip(df["cost"], 0, clip)

    _, (ax1, ax2, ax3, ax4) = plt.subplots(
        1,
        4,
        figsize=(10, 3),
        sharey=True,
        gridspec_kw={"wspace": 0.05, "width_ratios": [0.3, 0.7, 0.7, 0.3]},
    )

    df_burnin["cost"].plot.hist(bins=30, log=True, ax=ax1, orientation="horizontal")
    ax1.set_xlim(ax1.get_xlim()[::-1])
    for _, _df in df_burnin.groupby(["emodel", "chain_id"]):
        ax2.plot(_df.cost.to_numpy(), "0.5", lw=0.2, zorder=0)
    for _, _df in df_burnin.groupby(["emodel", "chain_id"]):
        ax2.plot(0, _df.cost.to_numpy()[0], ".r", ms=2)
        ax2.plot(len(_df.cost.to_numpy()) - 1, _df.cost.to_numpy()[-1], ".b", ms=2)

    ax2.plot(0, _df.cost.to_numpy()[0], ".r", ms=2, label="initial points")
    ax2.plot(
        len(_df.cost.to_numpy()) - 1, _df.cost.to_numpy()[-1], ".b", ms=2, label="final points"
    )
    ax2.legend()
    for _, _df in df.groupby(["emodel", "chain_id"]):
        ax3.plot(_df.cost.to_numpy(), "0.5", lw=0.2, zorder=0)
    for _, _df in df.groupby(["emodel", "chain_id"]):
        ax3.plot(0, _df.cost.to_numpy()[0], ".r", ms=3)
        ax3.plot(len(_df.cost.to_numpy()) - 1, _df.cost.to_numpy()[-1], ".b", ms=3)
    df["cost"].plot.hist(bins=30, log=True, ax=ax4, orientation="horizontal")

    ax2.set_rasterization_zorder(1)
    ax3.set_rasterization_zorder(1)
    ax1.set_xlabel("# models")
    ax2.set_xlabel("accepted iteration")
    ax3.set_xlabel("accepted iteration")
    ax4.set_xlabel("# models")
    ax1.set_ylabel("cost")
    plt.tight_layout()
    plt.savefig(filename)


def plot_cost_convergence(df, filename="cost_convergence.png"):
    """Plot the value of the cost of each chain as a function of iteration."""
    plt.figure(figsize=(5, 3))
    for emodel in df.emodel.unique():
        _df = df[df.emodel == emodel]
        for chain_id in _df.chain_id.unique():
            plt.plot(_df[_df.chain_id == chain_id].cost.to_numpy(), "0.5", lw=0.2)
    for emodel in df.emodel.unique():
        _df = df[df.emodel == emodel]
        for chain_id in _df.chain_id.unique():
            plt.plot(0, _df[_df.chain_id == chain_id].cost.to_numpy()[0], ".r", ms=3)
            plt.plot(
                len(_df[_df.chain_id == chain_id].cost.to_numpy()) - 1,
                _df[_df.chain_id == chain_id].cost.to_numpy()[-1],
                ".b",
                ms=3,
            )
        plt.plot(0, _df[_df.chain_id == chain_id].cost.to_numpy()[0], ".r", ms=3)
        plt.plot(
            len(_df[_df.chain_id == chain_id].cost.to_numpy()) - 1,
            _df[_df.chain_id == chain_id].cost.to_numpy()[-1],
            ".b",
            ms=3,
        )
    plt.legend(loc="best")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.tight_layout()
    plt.savefig(filename)


def dpoint2pointcloud(X, i, metric):
    """
    Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points

    Copied from ripser.py

    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of data
    i: int
        The index of the point from which to return all distances
    metric: string or callable
        The metric to use when calculating distance between instances in a
        feature array
    """

    ds = pairwise_distances(X, X[i, :][None, :], metric=metric).flatten()
    ds[i] = 0
    return ds


def get_greedy_perm(X, n_perm=None, dist_matrix=False, metric="euclidean"):
    """
    Compute a furthest point sampling permutation of a set of points

    Copied from ripser.py

    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix
    dist_matrix: bool
        Indicator that X is a distance matrix, if not we compute
        distances in X using the chosen metric.
    n_perm: int
        Number of points to take in the permutation
    metric: string or callable
        The metric to use when calculating distance between instances in a
        feature array
    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation
    lambdas: ndarray(n_perm)
        Covering radii at different points
    dperm2all: ndarray(n_perm, n_samples)
        Distances from points in the greedy permutation to points
        in the original point set
    """
    if not n_perm:
        n_perm = X.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)

    def dpoint2all(i):
        if dist_matrix:
            return X[i, :]
        else:
            return dpoint2pointcloud(X, i, metric)

    ds = dpoint2all(0)
    dperm2all = [ds]
    for i in tqdm(range(1, n_perm)):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(idx))
        ds = np.minimum(ds, dperm2all[-1])
    lambdas[-1] = np.max(ds)
    dperm2all = np.array(dperm2all)
    return (idx_perm, lambdas, dperm2all)


def plot_feature_correlations(df, split, pearson_thresh=0.6, figure_name="feature_corrs.pdf"):
    """Plot feature correlations."""
    _df = df[df.cost < split]
    c = _df["cost"]
    _df = _df["scores"]
    with PdfPages(figure_name) as pdf:
        for col1, col2 in itertools.combinations(_df.columns, 2):
            x, y = _df[col1].to_list(), _df[col2].to_list()
            p = pearsonr(x, y)[0]
            if abs(p) > pearson_thresh:
                plt.figure(figsize=(5, 4))
                plt.scatter(x, y, marker=".", s=0.5, c=c, rasterized=True)
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.suptitle(p)
                plt.colorbar(label="cost", shrink=0.5)
                pdf.savefig()
                plt.close()


def _autocorr(series):
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    return x, y


def plot_autocorrelation(df, n_emodels=10, filename="autocorrelation.pdf"):
    """Autocorrelation plot, adapted from pandas.plotting.autocorrelation_plot."""
    plt.figure(figsize=(5, 3))
    ax = plt.gca()
    for emodel in df.emodel.unique()[:n_emodels]:
        Y = []
        _df = df[df.emodel == emodel]
        for _, s in _df["parameters"].T.iterrows():
            x, y = _autocorr(s)
            Y.append(y)
            ax.plot(x, y, c="0.5", lw=0.1, zorder=-1)
        ax.plot(x, np.mean(Y, axis=0), c="C0")

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    n = len(_df)
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="k", lw=0.5)
    ax.axhline(y=z95 / np.sqrt(n), color="k", lw=0.5)
    ax.axhline(y=0.0, color="k", lw=0.5)
    ax.axhline(y=-z95 / np.sqrt(n), color="k", lw=0.5)
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="k", lw=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(0, int(0.8 * n))
    plt.tight_layout()
    plt.savefig(filename)


def get_mean_sd(efeatures, feat):
    """Get experimenatl mean and sd."""
    f_mean, f_std = 0, 0
    if "efeatures" in efeatures:
        for f in efeatures["efeatures"]:
            try:
                _f = ".".join([f["protocol_name"], f["recording_name"], f["efel_feature_name"]])
            except TypeError:
                _f = "fail"
            if _f == feat:
                f_mean = f["mean"]
                f_std = f["original_std"]

    else:
        feat_split = feat.split(".")
        if feat_split[0] == "RinProtocol":
            feat_split[0] = "Rin"
        if feat_split[0] == "RMPProtocol":
            feat_split[0] = "RMP"
            feat_split[3] = "voltage_base"
        if feat_split[0] == "SearchHoldingCurrent":
            feat_split[0] = "RinHoldCurrent"
            if feat_split[3] == "steady_state_voltage_stimend":
                feat_split[0] = "Rin"
                feat_split[3] = "voltage_base"
        if feat_split[0] == "SearchThresholdCurrent":
            feat_split[0] = "Threshold"
        for f in efeatures[feat_split[0]][".".join(feat_split[1:3])]:
            if f["feature"] == feat_split[3]:
                f_mean = f["val"][0]
                f_std = f["val"][1]
    return f_mean, f_std


def plot_reduced_feature_distributions(
    df, emodel, access_point, features=None, filename="reduced_feature_distributions.pdf"
):
    """Plot feature distributions of some features with violin plots."""
    if features is None:
        features = df.features.columns
    n_feat = len(features)
    fig = plt.figure(figsize=(5, 1.5 * n_feat), constrained_layout=True)
    efeatures = access_point.get_json(emodel, "features")
    gs = fig.add_gridspec(n_feat, 1)
    for i, feature in enumerate(features):
        if feature != "mtype":
            ax = fig.add_subplot(gs[i, 0], axes_class=axisartist.Axes)
            ax.axis["top", "left", "right"].set_visible(False)
            plt.xlabel(ALL_LABELS.get(feature, feature))
            sns.violinplot(data=df[[("features", feature)]], ax=ax, bwr=0.02, orient="h")
            f_mean, f_std = get_mean_sd(efeatures, feature)
            plt.axvline(f_mean, c="k", label="exp. mean")
            plt.axvline(f_mean - f_std, c="k", ls="--")
            plt.axvline(f_mean + f_std, c="k", ls="--", label="exp. 1sd")
            plt.axvline(f_mean - 2 * f_std, c="k", ls="-.")
            plt.axvline(f_mean + 2 * f_std, c="k", ls="-.", label="exp. 2sd")
            plt.axvline(f_mean - 5 * f_std, c="k", ls="dotted")
            plt.axvline(f_mean + 5 * f_std, c="k", ls="dotted", label="exp. 5sd")

    plt.tight_layout()
    plt.savefig(filename)


def plot_feature_distributions(
    df, emodel, access_point, filename="feature_distributions.pdf", log_scale=True
):
    """Plot feature distributions."""
    efeatures = access_point.get_json(emodel, "features")
    with PdfPages(filename) as _pdf:
        feat_df = df["features"]
        for feat in feat_df.columns:
            if feat != "mtype":
                f_mean, f_std = get_mean_sd(efeatures, feat)
                plt.figure(figsize=(5, 3))
                plt.hist(
                    np.clip(feat_df[feat], f_mean - 5 * f_std, f_mean + 5 * f_std),
                    bins=100,
                    histtype="step",
                    log=log_scale,
                )
                plt.axvline(f_mean, c="k", label="exp. mean")
                plt.axvline(f_mean - f_std, c="k", ls="--")
                plt.axvline(f_mean + f_std, c="k", ls="--", label="exp. 1sd")
                plt.axvline(f_mean - 2 * f_std, c="k", ls="-.")
                plt.axvline(f_mean + 2 * f_std, c="k", ls="-.", label="exp. 2sd")
                plt.axvline(f_mean - 5 * f_std, c="k", ls="dotted")
                plt.axvline(f_mean + 5 * f_std, c="k", ls="dotted", label="exp. 5sd")

                plt.xlabel(ALL_LABELS.get(feat, feat))
                plt.ylabel("# combos")
                plt.legend()
                plt.tight_layout()
                _pdf.savefig()
                plt.close()


def bin_data(p1, p2, f, n=20, mode="mean", _min1=-1.0, _max1=1.0, _min2=-1.0, _max2=1.0):
    """Bin data to make heatmap."""
    _df = pd.DataFrame()
    _df["p1"] = np.array(n * (p1 - _min1) / (_max1 - _min1 + 1e-10), dtype=int)
    _df["p2"] = np.array(n * (p2 - _min2) / (_max2 - _min2 + 1e-10), dtype=int)
    _df["f"] = f
    _df = getattr(_df.groupby(["p1", "p2"]), mode)().reset_index()
    m = np.zeros([n + 1, n + 1])
    m[_df["p1"], _df["p2"]] = _df["f"]
    m[m == 0] = np.nan
    return m[:-1][:, :-1]


def _get_2d_data(df, feature, param1, param2, n_bins=20, perc=10):
    """Bin data to make heatmap."""
    p1 = df[("normalized_parameters", param1)].to_numpy()
    p2 = df[("normalized_parameters", param2)].to_numpy()

    if feature is None:
        m = bin_data(p1, p2, np.ones(len(p1)), n=n_bins, mode="sum")
    else:
        m = bin_data(p1, p2, feature, n=n_bins, mode="mean")
    return np.clip(
        m, np.percentile(m[~np.isnan(m)], perc), np.percentile(m[~np.isnan(m)], 100 - perc)
    )


def _plot_2d_data(ax, m, vmin, vmax, rev=True, cmap="gnuplot", normalize=False):
    """Plot heatmap."""
    dx = 1.0 / (len(m) - 1.0)
    return ax.imshow(
        m.T / np.nanmax(np.nanmax(m)) if normalize else m.T,
        origin="lower",
        aspect="auto",
        extent=(-1 - dx, 1 + dx, -1 - dx, 1 + dx),
        cmap=f"{cmap}_r" if rev else cmap,
        interpolation="nearest",
        vmin=0 if normalize else vmin,
        vmax=1 if normalize else vmax,
    )


def plot_corner(
    df,
    feature=None,
    filename="corner.pdf",
    n_bins=20,
    cmap="gnuplot",
    normalize=False,
    highlights=None,
):
    """Make a corner plot which consists of scatter plots of all pairs.

    Args:
        feature (str): name of feature for coloring heatmap
        filename (str): name of figure for corner plot
    """
    params = np.array(sorted(df.normalized_parameters.columns.to_list()))
    _params = np.array([PARAM_LABELS.get(p, p) for p in params])
    params = params[np.argsort(_params)]
    n_params = len(params)

    # get feature data
    _feature = None
    if feature is not None:
        _feature = df[feature].to_numpy()

        if np.std(_feature) < 1e-5:
            print("no data to plot")
            return None

    # precompute heatmaps to get a global vmin/vmax
    m = []
    vmin = 1e10
    vmax = -1e10
    for i, param1 in enumerate(params):
        m.append([])
        for j, param2 in enumerate(params):
            if j < i:
                _m = _get_2d_data(df, _feature, param2, param1, n_bins=n_bins)
                m[i].append(_m)
                vmin = min(vmin, min(_m[~np.isnan(_m)].flatten()))  # pylint: disable=nested-min-max
                vmax = max(vmax, max(_m[~np.isnan(_m)].flatten()))  # pylint: disable=nested-min-max
    fig = plt.figure(figsize=(5 + 0.5 * n_params, 5 + 0.5 * n_params))
    gs = fig.add_gridspec(n_params, n_params, hspace=0.1, wspace=0.1)
    im = None
    for i, param1 in enumerate(params):
        _param1 = PARAM_LABELS.get(param1, param1)
        for j, param2 in enumerate(params):
            _param2 = PARAM_LABELS.get(param2, param2)
            ax = plt.subplot(gs[i, j])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            if j >= i + 1:
                ax.set_frame_on(False)
            elif j < i:
                ax.set_frame_on(True)
                im = _plot_2d_data(
                    ax, m[i][j], vmin, vmax, rev=feature is not None, cmap=cmap, normalize=normalize
                )
                if highlights is not None:
                    for _i, _ in zip(*highlights):
                        plt.scatter(
                            df.loc[_i, ("normalized_parameters", param2)],
                            df.loc[_i, ("normalized_parameters", param1)],
                            c="g",
                            s=50,
                        )
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
            else:
                if _feature is None:
                    ax.hist(
                        df[("normalized_parameters", param1)].to_numpy(),
                        bins=n_bins,
                        color="k",
                        histtype="step",
                    )
                    ax.set_xlim(-1, 1)
                    ax.set_frame_on(True)
                else:
                    ax.set_frame_on(False)

            if j == 0:
                ax.set_ylabel(_param1, rotation="horizontal", horizontalalignment="right")
            if i == n_params - 1:
                ax.set_xlabel(_param2, rotation="vertical")
            if j == 0 and i == 0:
                ax.set_ylabel(_param1, rotation="horizontal", horizontalalignment="left")
            if (i == j) if _feature is None else (i == j + 1):
                ax.set_ylabel(_param1, rotation="horizontal", horizontalalignment="left")
                ax.yaxis.set_label_position("right")

    if im is not None:
        if n_params > 6:
            axs = [
                plt.subplot(gs[i + 3, n_params - j - 1])
                for i in range(n_params - 6)
                for j in range(3)
            ]
        else:
            axs = [plt.subplot(gs[-1, -1])]
        plt.colorbar(
            im,
            orientation="vertical",
            ax=axs,
            label=feature[1] if feature is not None else "number of models",
        )
    plt.tight_layout()
    plt.savefig(filename)

    return fig


def _top_params(cor, sd=5):
    cor_n = cor.to_numpy()
    if np.shape(cor_n)[0] == np.shape(cor_n)[1]:
        _x = np.triu(cor, 1).flatten()
        _x = _x[abs(_x) > 0]
        thresh = sd * np.std(_x)
    else:
        thresh = sd * np.std(cor_n.flatten())
    if sd > 0:
        _cor = cor[cor > thresh]
    else:
        _cor = cor[cor < thresh]
    tuples = []
    for x in cor.index:
        for y in cor.columns:
            if not np.isnan(_cor.loc[x, y]):
                if (y, x) not in tuples:
                    tuples += [(x, y)]
    return tuples


def plot_best_corr(df, cor, x_col, y_col, filename, sd=5):
    """Plot only highest correlated tuples."""
    tuples = _top_params(cor, sd)
    with PdfPages(filename) as pdf:
        for x, y in tqdm(tuples):
            x_data = df[(x_col, x)]
            y_data = df[(y_col, y)]

            plt.figure(figsize=(4, 3))
            plt.scatter(x_data, y_data, c=df["cost"], s=0.2, marker=".", rasterized=True)
            plt.suptitle(f"MI: {cor.loc[x, y]}")
            plt.colorbar()
            plt.xlabel(ALL_LABELS.get(x, x))
            plt.ylabel(ALL_LABELS.get(y, y))
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def plot_MI(MI, with_cluster=False):
    """Plot MI matrix."""
    if with_cluster:
        sorted_labels = cluster_matrix(abs(MI))
        MI = MI.loc[sorted_labels, sorted_labels]
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    _MI = MI.copy()
    _MI.index = [ALL_LABELS.get(p, p) for p in MI.index]
    _MI.columns = [ALL_LABELS.get(p, p) for p in MI.columns]
    if MI.min().min() >= 0:
        vmin = 0
        vmax = None
        cmap = "viridis"
    else:
        vmax = MI.abs().max().max()
        vmin = -vmax
        cmap = "bwr"

    sns.heatmap(
        data=_MI,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        linewidths=0.5,
        linecolor="k",
        cbar_kws={"label": "MI", "shrink": 0.3},
        xticklabels=True,
        yticklabels=True,
        square=True,
    )
    plt.tight_layout()


def get_2d_correlations(
    df,
    x_col="normalized_parameters",
    y_col="normalized_parameters",
    mi_max=1,
    feature=None,
    tpe="MI",
):
    """Get 2d correlations."""
    if x_col == y_col:
        tuples = itertools.combinations(df[x_col].columns, 2)
    else:
        tuples = itertools.product(df[x_col].columns, df[y_col].columns)

    MI = pd.DataFrame(index=df[x_col].columns, columns=df[y_col].columns, dtype=float)
    for x, y in tuples:
        if feature is None:
            if df[(x_col, x)].std() == 0 or df[(y_col, y)].std() == 0:
                mi = 0
            else:
                if tpe == "MI":
                    mi = mi_gaussian(
                        np.vstack([df[(x_col, x)].to_numpy(), df[(y_col, y)].to_numpy()]).T
                    )
                if tpe == "pearson":
                    mi = pearsonr(df[(x_col, x)].to_numpy(), df[(y_col, y)])[0]
        else:
            feat = df[("features", feature)].to_numpy()
            if feat.std() > 0:
                feat = (feat - feat.mean()) / feat.std()
            else:
                feat = 0 * feat
            if tpe == "MI":
                mi = rsi_gaussian(
                    np.vstack([feat, df[(x_col, x)].to_numpy(), df[(y_col, y)].to_numpy()]).T
                )
            if tpe == "pearson":
                mi = pearsonr(feat, [df[(x_col, x)].to_numpy(), df[(y_col, y)].to_numpy()])[0]
        if tpe == "MI":
            mi = min(mi, mi_max)
        if np.isnan(mi):
            mi = 0
        MI.loc[x, y] = mi
        if x_col == y_col:
            MI.loc[y, x] = mi
            MI.loc[x, x] = 0.0
            MI.loc[y, y] = 0.0
    return MI


def plot_top_corner(df, out_path):
    """Plot subcorner of top correlations."""
    for feat in df["features"].columns:
        cor = get_2d_correlations(df, feature=feat)
        plot_MI(cor)
        plt.savefig(out_path / f"RSI_param_param_{feat}.pdf")
        tuples = _top_params(cor, sd=-5)
        params = set(np.array(tuples).flatten())
        _df = df.drop(
            columns=[
                c for c in df.columns if (c[0] == "normalized_parameters") and (c[1] not in params)
            ]
        )
        plot_corner(
            _df,
            feature=("features", feat),
            filename=out_path / f"RSI_top_corner_{feat}.pdf",
        )
        plt.close()
