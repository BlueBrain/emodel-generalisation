"""Main workflow to generalise electrical models using MCMC.

See example/workflow for an example on how to run it.

TODO: improve docstrings.
"""

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
import os
import shutil
from functools import partial
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml

try:
    from emodeldb.config_tools import pull_config  # pylint: disable=import-error
except ImportError:
    pass
from luigi_tools.target import OutputLocalTarget
from luigi_tools.task import WorkflowTask
from luigi_tools.task import WorkflowWrapperTask
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.adaptation import adapt_soma_ais
from emodel_generalisation.adaptation import build_all_resistance_models
from emodel_generalisation.adaptation import find_rho_factors
from emodel_generalisation.adaptation import make_evaluation_df
from emodel_generalisation.exemplars import generate_exemplars
from emodel_generalisation.mcmc import filter_features
from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import plot_autocorrelation
from emodel_generalisation.mcmc import plot_corner
from emodel_generalisation.mcmc import plot_feature_distributions
from emodel_generalisation.mcmc import plot_full_cost_convergence
from emodel_generalisation.mcmc import plot_reduced_feature_distributions
from emodel_generalisation.mcmc import run_several_chains
from emodel_generalisation.mcmc import save_selected_emodels
from emodel_generalisation.model.evaluation import evaluate_rho
from emodel_generalisation.model.evaluation import evaluate_rho_axon
from emodel_generalisation.model.evaluation import feature_evaluation
from emodel_generalisation.model.modifiers import synth_axon
from emodel_generalisation.model.modifiers import synth_soma
from emodel_generalisation.morph_utils import create_combos_df
from emodel_generalisation.morph_utils import rediametrize
from emodel_generalisation.select import select_valid
from emodel_generalisation.tasks.utils import EmodelAwareTask
from emodel_generalisation.tasks.utils import EmodelLocalTarget
from emodel_generalisation.tasks.utils import ParallelTask
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_score_df
from emodel_generalisation.utils import plot_traces
from emodel_generalisation.validation import compare_adaptation
from emodel_generalisation.validation import plot_adaptation_summaries
from emodel_generalisation.validation import validate_morphologies

# pylint: disable=too-many-nested-blocks,too-many-lines,too-many-function-args,too-many-branches


class GetEmodelConfig(WorkflowTask):
    """Get config folder with emodel setting via emodeldb (not OS) or locally.

    If mode =='local', use 'config_path' for the config folder, and 'mechanisms_path' for the
    mechanisms folder to compile, and 'generalisation_rule' with filters for morphologies
    generalisation.
    """

    emodel = luigi.Parameter()
    mode = luigi.ChoiceParameter(default="local", choices=["local", "emodeldb"])

    config_path = luigi.Parameter(default="configs")
    mechanisms_path = luigi.Parameter(default="mechanisms")
    generalisation_rule_path = luigi.Parameter(default="generalisation_rule.yaml")

    out_config_path = luigi.Parameter(default="configs")
    emodeldb_version = luigi.Parameter(default="proj38")

    def run(self):
        """ """
        if self.mode == "emodeldb":
            data = pull_config(
                self.emodel, version=self.emodeldb_version, to=self.output().pathlib_path.parent
            )
            with open(self.output().pathlib_path.parent / "generalisation_rule.yaml", "w") as f:
                yaml.dump(data["generalisation"], f)
            print("Fetching emodel:", data)
            os.popen(f"nrnivmodl {self.output().pathlib_path.parent / 'mechanisms'}").read()

        if self.mode == "local":
            shutil.copytree(self.config_path, self.output().pathlib_path)
            os.popen(f"nrnivmodl {self.mechanisms_path}").read()
            shutil.copy(
                self.generalisation_rule_path,
                self.output().pathlib_path.parent / "generalisation_rule.yaml",
            )

    def output(self):
        """ """
        return EmodelLocalTarget(self.out_config_path)


class CreateComboDF(EmodelAwareTask, WorkflowTask):
    """Create dataframe with combos mixing mtype, etype and emodels."""

    emodel = luigi.Parameter()
    morphology_dataset_path = luigi.Parameter(default="dataset.csv")
    combodf_path = luigi.Parameter(default="combo_df.csv")
    n_morphs = luigi.IntParameter(default=None)
    n_min_per_mtype = luigi.IntParameter(default=10)

    def requires(self):
        """ """
        return GetEmodelConfig(emodel=self.emodel)

    def run(self):
        """ """
        combos_df = create_combos_df(
            self.morphology_dataset_path,
            self.input().pathlib_path.parent / "generalisation_rule.yaml",
            self.emodel,
            self.n_min_per_mtype,
            self.n_morphs,
        )
        combos_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return EmodelLocalTarget(self.combodf_path)


class ReDiametrize(WorkflowTask):
    """Rediametrize morphologies with diameter model from exemplar."""

    emodel = luigi.Parameter()
    morphology_folder = luigi.Parameter(default="rediametrized_morphologies")
    rediametrized_combo_df_path = luigi.Parameter(default="rediametrized_combo_df.csv")
    diameter_model_path = luigi.Parameter(default="diameter_model.yaml")
    mode = luigi.Parameter("simple")

    def requires(self):
        """ """
        return CreateComboDF(emodel=self.emodel)

    def run(self):
        """ """
        combo_df = pd.read_csv(self.input().path)
        combo_df = rediametrize(
            combo_df,
            self.output().pathlib_path.parent,
            self.diameter_model_path,
            self.morphology_folder,
        )
        combo_df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return EmodelLocalTarget(self.rediametrized_combo_df_path)


class CreateExemplar(WorkflowTask):
    """Create the exemplar morphology for an emodel and a population of morphologies."""

    emodel = luigi.Parameter()
    exemplar_model_path = luigi.Parameter(default="exemplar_models.yaml")
    surface_percentile = luigi.FloatParameter(default=25)

    exemplar_path = luigi.Parameter(default=None)

    def requires(self):
        """ """
        return {
            "emodel_config": GetEmodelConfig(emodel=self.emodel),
            "combodf": ReDiametrize(emodel=self.emodel),
        }

    def run(self):
        """ """
        combo_df = pd.read_csv(self.input()["combodf"].path)
        exemplar_data = generate_exemplars(
            combo_df,
            figure_folder=self.output().pathlib_path.parent / "exemplar_figures",
            surface_percentile=self.surface_percentile,
        )

        if self.exemplar_path is not None:
            exemplar_data["paths"]["all"] = self.exemplar_path

        with open(self.output().path, "w") as f:
            yaml.dump(exemplar_data, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.exemplar_model_path)


class RunMCMCBurnIn(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Run burn-in phase MCMC exploration for an emodel."""

    emodel = luigi.Parameter()
    mcmc_df_path = luigi.Parameter(default="mcmc_df_burnin.csv")
    temperature = luigi.FloatParameter(default=100.0)
    n_steps = luigi.IntParameter(default=20)
    n_chains = luigi.IntParameter(default=80)
    proposal_std = luigi.FloatParameter(default=0.1)

    def requires(self):
        """ """
        return {
            "emodel_config": GetEmodelConfig(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())

        access_point = self.get_access_point(with_seeds=False)
        access_point.morph_path = exemplar_data["paths"]["all"]

        access_point.morph_modifiers = [
            partial(synth_axon, params=exemplar_data["ais"]["popt"], scale=1.0)
        ]
        access_point.morph_modifiers.insert(
            0, partial(synth_soma, params=exemplar_data["soma"], scale=1.0)
        )

        frozen_params = None
        if Path("frozen_parameters.json").exists():
            with open("frozen_parameters.json", "r") as f:
                frozen_params = json.load(f)
            print("Using frozen parameters.")

        run_several_chains(
            proposal_params={"type": "normal", "std": self.proposal_std},
            temperature=self.temperature,
            n_steps=self.n_steps,
            n_chains=self.n_chains,
            access_point=access_point,
            emodel=self.emodel,
            run_df_path=self.output().path,
            results_df_path=self.output().pathlib_path.parent / "chains",
            parallel_lib=self.parallel_factory,
            random_initial_parameters=True,
            mcmc_log_file=self.output().pathlib_path.parent / "mcmc_log.txt",
            frozen_params=frozen_params,
        )

    def complete(self):
        """ """
        return WorkflowTask.complete(self)

    def output(self):
        """ """
        return EmodelLocalTarget(self.mcmc_df_path)


class RunMCMC(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Run MCMC exploration for an emodel using best emodels from burn-in phase for speed up."""

    emodel = luigi.Parameter()
    mcmc_df_path = luigi.Parameter(default="mcmc_df.csv")
    temperature = luigi.FloatParameter(default=100.0)
    n_steps = luigi.IntParameter(default=500)
    n_chains = luigi.IntParameter(default=80)
    proposal_std = luigi.FloatParameter(default=0.1)
    split_perc = luigi.FloatParameter(default=10)
    resume = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        return {
            "emodel_config": GetEmodelConfig(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
            "mcmc_burnin": RunMCMCBurnIn(emodel=self.emodel),
        }

    def run(self):
        """ """
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        if self.resume and not Path(self.output().path).exists():
            print("we cannot resume without a previous mcmc run")
            self.resume = False

        if self.resume:
            mcmc_df = load_chains(self.output().path, with_single_origin=False)
            ids = [_df.index[-1] for _, _df in mcmc_df.groupby("emodel")]
            final_path = self.output().pathlib_path.parent / "final_mcmc_resume.json"
        else:
            mcmc_df = (
                load_chains(self.input()["mcmc_burnin"].path, with_single_origin=False)
                .drop_duplicates()
                .reset_index(drop=True)
            )
            ids = mcmc_df.sort_values(by="cost").head(self.n_chains).index
            final_path = self.output().pathlib_path.parent / "final_mcmc_burnin.json"
        save_selected_emodels(mcmc_df, ids, emodel=self.emodel, final_path=final_path)

        access_point = self.get_access_point(final_path=final_path)
        access_point.morph_path = exemplar_data["paths"]["all"]

        access_point.morph_modifiers = [
            partial(synth_axon, params=exemplar_data["ais"]["popt"], scale=1.0)
        ]
        access_point.morph_modifiers.insert(
            0, partial(synth_soma, params=exemplar_data["soma"], scale=1.0)
        )
        frozen_params = None
        if Path("frozen_parameters.json").exists():
            with open("frozen_parameters.json", "r") as f:
                frozen_params = json.load(f)
            print("Using frozen parameters.")

        run_several_chains(
            proposal_params={"type": "normal", "std": self.proposal_std},
            temperature=self.temperature,
            n_steps=self.n_steps,
            n_chains=1,
            access_point=access_point,
            emodel=None,
            run_df_path=self.output().path,
            results_df_path=self.output().pathlib_path.parent / "chains",
            parallel_lib=self.parallel_factory,
            random_initial_parameters=False,
            mcmc_log_file=self.output().pathlib_path.parent / "mcmc_log.txt",
            chain_df=self.output().path if self.resume else None,
            frozen_params=frozen_params,
        )

    def complete(self):
        """ """
        if self.resume:
            return False
        return WorkflowTask.complete(self)

    def output(self):
        """ """
        return EmodelLocalTarget(self.mcmc_df_path)


class PlotMCMCResults(EmodelAwareTask, WorkflowTask):
    """Make various plots of MCMC results."""

    emodel = luigi.Parameter()
    mcmc_figures_path = luigi.Parameter(default="mcmc_figures")
    split = luigi.FloatParameter(default=5)
    max_split = luigi.FloatParameter(default=10)

    def requires(self):
        """ """
        return {"mcmc": RunMCMC(emodel=self.emodel), "burnin": RunMCMCBurnIn(emodel=self.emodel)}

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(parents=True)

        burnin_mcmc_df = load_chains(self.input()["burnin"].path, with_single_origin=False)
        burnin_mcmc_df["cost"] = np.clip(burnin_mcmc_df["cost"], 0, 20)

        mcmc_df = pd.read_csv(self.input()["mcmc"].path)

        # plot_step_size(mcmc_df, self.output().pathlib_path / "stepsize.pdf")

        mcmc_df = load_chains(mcmc_df, with_single_origin=False)

        plot_autocorrelation(mcmc_df, filename=self.output().pathlib_path / "autocorrelation.pdf")

        access_point = self.get_access_point()
        plot_feature_distributions(
            mcmc_df[mcmc_df.cost < self.split],
            self.emodel,
            access_point,
            filename=self.output().pathlib_path / "feature_distributions.pdf",
        )

        _df = filter_features(mcmc_df[mcmc_df.cost < self.split])
        plot_reduced_feature_distributions(
            _df,
            self.emodel,
            access_point,
            filename=self.output().pathlib_path / "feature_distributions_reduced.pdf",
        )

        plot_full_cost_convergence(
            burnin_mcmc_df,
            mcmc_df,
            clip=self.max_split,
            filename=self.output().pathlib_path / "cost_convergence.pdf",
        )

        mcmc_df = mcmc_df[mcmc_df.cost < self.max_split].reset_index(drop=True)

        # plot the number of emodel for which the given feature is the worst (score=cost)
        max_feat = mcmc_df["scores"].idxmax(axis=1).value_counts(ascending=True)
        plt.figure(figsize=(7, 9))
        max_feat.plot.barh(ax=plt.gca())
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(self.output().pathlib_path / "worst_features.pdf")

        # plot_feature_correlations(
        #    mcmc_df, self.split, figure_name=self.output().pathlib_path / "feature_corrs.pdf"
        # )

        # plot_parameter_distributions(
        #    mcmc_df, self.split, filename=self.output().pathlib_path / "parameters.pdf"
        # )
        # plot_score_distributions(
        #    mcmc_df, self.split, filename=self.output().pathlib_path / "scores.pdf"
        # )

        mcmc_df = mcmc_df[mcmc_df.cost < self.split].reset_index(drop=True)
        plot_corner(
            mcmc_df.reset_index(drop=True),
            feature=None,
            filename=self.output().pathlib_path / "corner.pdf",
        )

        plot_corner(
            mcmc_df.reset_index(drop=True),
            feature="cost",
            filename=self.output().pathlib_path / "corner_cost.pdf",
        )
        plt.close("all")

        # plot corner for each feature starting from the worst
        for feature in mcmc_df["scores"].mean(axis=0).sort_values(ascending=False).index:
            print("corner plot of ", feature)
            plot_corner(
                mcmc_df.reset_index(drop=True),
                feature=("scores", feature),
                filename=self.output().pathlib_path / f"corner_{feature}.pdf",
            )
            plt.close()

    def output(self):
        """ """
        return EmodelLocalTarget(self.mcmc_figures_path)


class SelectRobustParams(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Select a small set of robust parameters."""

    emodel = luigi.Parameter()

    split = luigi.FloatParameter(default=5)
    robust_emodels_path = luigi.Parameter(default="final.json")
    n_emodels = luigi.IntParameter(default=10)

    def requires(self):
        """ """
        return {
            "mcmc": RunMCMC(emodel=self.emodel),
            "emodel_config": GetEmodelConfig(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        mcmc_df = load_chains(self.input()["mcmc"].path)
        mcmc_df = mcmc_df[mcmc_df.cost < self.split].reset_index(drop=True)

        # if we have restarted, we rename emodels to ensure we don't have duplicates
        ids = mcmc_df.sample(self.n_emodels, random_state=42).index
        for gid in ids:
            mcmc_df.loc[gid, "emodel"] = f"{self.emodel}_{gid}"

        mcmc_df.loc[ids].to_csv(self.output().pathlib_path.parent / "emodels.csv")
        save_selected_emodels(mcmc_df, ids, emodel=self.emodel, final_path=self.output().path)

        df = mcmc_df.loc[ids, "normalized_parameters"]
        df = df.melt(ignore_index=False)
        df["emodel"] = df.index

        plt.figure(figsize=(6, 20))
        sns.stripplot(data=df, x="value", y="variable", ax=plt.gca())
        plt.tight_layout()
        plt.savefig(self.output().pathlib_path.parent / "emodel_parameters.pdf")
        plt.close()

        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        df_traces = pd.DataFrame()
        with open(self.output().path) as f:
            emodels = list(json.load(f).keys())
        for i, emodel in enumerate(emodels):
            df_traces.loc[i, "path"] = exemplar_data["paths"]["all"]
            df_traces.loc[i, "name"] = Path(exemplar_data["paths"]["all"]).stem
            df_traces.loc[i, "ais_model"] = json.dumps(exemplar_data["ais"])
            df_traces.loc[i, "soma_model"] = json.dumps(exemplar_data["soma"])
            df_traces.loc[i, "ais_scaler"] = 1.0
            df_traces.loc[i, "soma_scaler"] = 1.0
            df_traces.loc[i, "emodel"] = emodel

        trace_folder = self.output().pathlib_path.parent / "robust_traces"
        trace_folder.mkdir(exist_ok=True)

        df_traces = feature_evaluation(
            df_traces,
            self.get_access_point(final_path=self.output().path),
            parallel_factory=self.parallel_factory,
            trace_data_path=trace_folder,
        )

        (self.output().pathlib_path.parent / "robust_traces_plots").mkdir(exist_ok=True)
        for emodel in df_traces.emodel.unique():
            plot_traces(
                df_traces[df_traces.emodel == emodel],
                trace_path=trace_folder,
                pdf_filename=self.output().pathlib_path.parent
                / "robust_traces_plots"
                / f"trace_{emodel}.pdf",
            )

    def output(self):
        """ """
        return EmodelLocalTarget(self.robust_emodels_path)


class RhoFactors(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Estimate the target rho value per me-types."""

    emodel = luigi.Parameter()
    rho_factors_path = luigi.Parameter(default="rhos_factors")

    def requires(self):
        """ """

        return {
            "exemplar": CreateExemplar(emodel=self.emodel),
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(exist_ok=True)
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        final_path = self.input()["emodel_parameters"].path
        with open(final_path) as f:
            emodels = json.load(f)

        access_point = self.get_access_point(final_path=final_path)
        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                print(f"Computing {mtype}")
                target_rhos = find_rho_factors(
                    emodels,
                    exemplar_data,
                    mtype,
                    access_point,
                    self.parallel_factory,
                    self.output().pathlib_path.parent,
                )
                with open(self.output().pathlib_path / f"rho_factors_{mtype}.yaml", "w") as f:
                    yaml.dump(target_rhos, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.rho_factors_path)


class ResistanceModels(EmodelAwareTask, WorkflowTask):
    """Constructs the AIS/soma input resistance models."""

    emodel = luigi.Parameter()
    resistance_model_path = luigi.Parameter(default="resistance_model.yaml")

    scale_min = luigi.FloatParameter(default=-1.0)
    scale_max = luigi.FloatParameter(default=1.0)
    scale_n = luigi.IntParameter(default=20)
    scale_lin = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        return {
            "emodel_config": GetEmodelConfig(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
        }

    def run(self):
        """ """
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        final_path = self.input()["emodel_parameters"].path

        final_path = self.input()["emodel_parameters"].path
        with open(final_path) as f:
            emodels = json.load(f)

        scales_params = {
            "min": self.scale_min,
            "max": self.scale_max,
            "n": self.scale_n,
            "lin": self.scale_lin,
        }

        access_point = self.get_access_point(final_path=final_path)

        models = build_all_resistance_models(
            access_point, emodels, exemplar_data, scales_params, self.output().pathlib_path.parent
        )
        with self.output().open("w") as f:
            yaml.dump(models, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.resistance_model_path)


class AdaptAisSoma(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Adapt AIS and Soma."""

    emodel = luigi.Parameter()
    n_steps = luigi.IntParameter(default=2)
    with_soma = luigi.BoolParameter(default=True)
    adapted_ais_soma_path = luigi.Parameter("adapted_soma_ais.csv")
    min_scale = luigi.FloatParameter(default=0.01)
    max_scale = luigi.FloatParameter(default=10.0)

    def requires(self):
        """ """
        return {
            "resistance": ResistanceModels(emodel=self.emodel),
            "combodb": ReDiametrize(emodel=self.emodel),
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
            "rhos": RhoFactors(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        resistance_models = yaml.safe_load(self.input()["resistance"].open())
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())

        final_path = self.input()["emodel_parameters"].path
        emodels = list(resistance_models["ais"].keys())
        combos_df = pd.read_csv(self.input()["combodb"].path)

        dfs = []
        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                print(f"Computing {mtype}")
                with open(self.input()["rhos"].pathlib_path / f"rho_factors_{mtype}.yaml") as f:
                    rhos = yaml.safe_load(f)
                df = make_evaluation_df(
                    combos_df[combos_df.mtype == mtype], emodels, exemplar_data, rhos
                )
                df = adapt_soma_ais(
                    df,
                    access_point=self.get_access_point(final_path=final_path),
                    models=resistance_models,
                    rhos=rhos,
                    parallel_factory=self.parallel_factory,
                    n_steps=self.n_steps,
                    min_scale=self.min_scale,
                    max_scale=self.max_scale,
                )

                df = evaluate_rho_axon(
                    df,
                    self.get_access_point(final_path=final_path),
                    parallel_factory=self.parallel_factory,
                )

                df = evaluate_rho(
                    df,
                    self.get_access_point(final_path=final_path),
                    parallel_factory=self.parallel_factory,
                )
                dfs.append(df)

        pd.concat(dfs).to_csv(self.output().path, index=False)

    def output(self):
        """ """
        return EmodelLocalTarget(self.adapted_ais_soma_path)


def _get_shap_feature_importance(shap_values):
    """From a list of shap values per folds, compute the global shap feature importance."""
    # average across folds
    mean_shap_values = np.mean(shap_values, axis=0)
    # average across labels
    if len(np.shape(mean_shap_values)) > 2:
        global_mean_shap_values = np.mean(mean_shap_values, axis=0)
        mean_shap_values = list(mean_shap_values)
    else:
        global_mean_shap_values = mean_shap_values

    # average across graphs
    shap_feature_importance = np.mean(abs(global_mean_shap_values), axis=0)
    return mean_shap_values, shap_feature_importance


class CreateMLRhoModels(WorkflowTask):
    """Create xgboost model of rhos factos as function of emodel parameter."""

    emodel = luigi.Parameter()
    rho_models_path = luigi.Parameter(default="rho_models")

    n_splits = luigi.IntParameter(default=10)
    n_repeats = luigi.IntParameter(default=5)

    def requires(self):
        """ """
        return {
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
            "rhos": RhoFactors(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(exist_ok=True)
        emodels_df = pd.read_csv(
            self.input()["emodel_parameters"].pathlib_path.parent / "emodels.csv",
            header=[0, 1],
            index_col=0,
        )
        emodels_df = emodels_df.set_index(("emodel", emodels_df["emodel"].columns[0]))
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                with open(self.input()["rhos"].pathlib_path / f"rho_factors_{mtype}.yaml") as f:
                    rhos = yaml.safe_load(f)
                param_bounds = {"rho": {}, "rho_axon": {}}
                for rho in ["rho", "rho_axon"]:
                    emodels = list(rhos[rho].keys())
                    df = emodels_df.loc[emodels]["normalized_parameters"]
                    df.loc[emodels, rho] = list(rhos[rho].values())

                    y = df[rho]
                    X = df.drop(columns=[rho])
                    params = []
                    for col in sorted(X.columns):
                        x = X[col]
                        p = pearsonr(x, y)[0]
                        if abs(p) > 0.4:
                            param_bounds[rho][col] = [
                                float(np.percentile(X[col], 10)),
                                float(np.percentile(X[col], 90)),
                            ]
                            params.append(col)
                            plt.figure(figsize=(4, 2.5))
                            plt.scatter(x, y)
                            plt.axvline(param_bounds[rho][col][0])
                            plt.axvline(param_bounds[rho][col][1])
                            plt.xlabel(ALL_LABELS[col])
                            plt.ylabel(rho)
                            plt.tight_layout()
                            plt.savefig(
                                self.output().pathlib_path / f"corr_{mtype}_{rho}_{col}.pdf"
                            )
                            plt.close()
                    if not param_bounds[rho]:
                        param_bounds[rho] = float(y.mean())
                    else:
                        X = X[params]
                        model = XGBRegressor(learning_rate=0.1)
                        folds = RepeatedKFold(
                            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=42
                        )
                        acc_scores = []
                        shap_values = []
                        for indices in tqdm(
                            folds.split(X, y=y), total=self.n_splits * self.n_repeats
                        ):
                            train_index, val_index = indices
                            model.fit(X.iloc[train_index], y.iloc[train_index])
                            acc_score = mean_absolute_error(
                                y.iloc[val_index], model.predict(X.iloc[val_index])
                            )
                            acc_scores.append(acc_score)
                            explainer = shap.TreeExplainer(model)
                            shap_value = explainer.shap_values(X)
                            shap_values.append(shap_value)

                        with open(self.output().pathlib_path / f"scores_{mtype}.txt", "a") as f:
                            print(rho, np.mean(acc_scores), np.std(acc_scores), file=f)

                        model.fit(X, y)
                        model.save_model(self.output().pathlib_path / f"model_{mtype}_{rho}.json")

                        shap_val, _ = _get_shap_feature_importance(shap_values)

                        X.columns = [ALL_LABELS.get(col, col) for col in X.columns]
                        shap.summary_plot(
                            shap_val,
                            X,
                            plot_type="bar",
                            max_display=5,
                            show=False,
                            plot_size=(5, 5),
                        )
                        plt.tight_layout()
                        plt.savefig(self.output().pathlib_path / f"bar_shap_{mtype}_{rho}.pdf")
                        plt.close()
                        shap.summary_plot(
                            shap_val,
                            X,
                            plot_type="dot",
                            max_display=5,
                            show=False,
                            color_bar_label="parameter value",
                            plot_size=(5, 3),
                        )
                        plt.tight_layout()
                        plt.savefig(self.output().pathlib_path / f"dot_shap_{mtype}_{rho}.pdf")
                        plt.close()

                with open(self.output().pathlib_path / f"param_bounds_{mtype}.yaml", "w") as f:
                    yaml.dump(param_bounds, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.rho_models_path)


class CreateMLGeneralisationModels(WorkflowTask):
    """Create an xgboost model for generalisable emodels."""

    emodel = luigi.Parameter()
    generalisation_models_path = luigi.Parameter(default="generalisation_models")
    n_splits = luigi.IntParameter(default=10)
    n_repeats = luigi.IntParameter(default=5)

    def requires(self):
        """ """
        return {
            "selected": SelectValidParameters(emodel=self.emodel),
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
            "evaluate": Evaluate(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(exist_ok=True)

        df_emodel = pd.read_csv(
            self.input()["emodel_parameters"].pathlib_path.parent / "emodels.csv",
            header=[0, 1],
            index_col=0,
        )

        params = [
            "gNaTgbar_NaTg.axonal",
            "gIhbar_Ih.somadend",
            "g_pas.all",
            "decay_CaDynamics_DC0.somatic",
        ]

        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                with open(self.input()["selected"].pathlib_path / f"selected_{mtype}.yaml") as f:
                    selected = yaml.safe_load(f)

                y = df_emodel["emodel"].isin(selected["emodels"]).astype(int)
                X = df_emodel["normalized_parameters"]
                # X = X[params]

                plt.figure(figsize=(5, 3))
                ps = df_emodel["parameters"]
                plt.scatter(ps[params[0]], ps[params[1]], c=ps[params[2]], label="good", marker="o")
                plt.colorbar(label=ALL_LABELS[params[2]], shrink=0.8)
                _ps = ps[y.to_numpy()[:, 0] < 1]
                plt.scatter(_ps[params[0]], _ps[params[1]], label="bad", marker="+", c="r")
                plt.xlabel(ALL_LABELS[params[0]])
                plt.ylabel(ALL_LABELS[params[1]])
                plt.tight_layout()
                plt.savefig(self.output().pathlib_path / f"{mtype}.pdf")
                plt.close()

                model = XGBClassifier(learning_rate=0.1)
                folds = RepeatedStratifiedKFold(
                    n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=42
                )
                acc_scores = []
                shap_values = []
                for indices in tqdm(folds.split(X, y=y), total=self.n_splits * self.n_repeats):
                    train_index, val_index = indices
                    model.fit(X.iloc[train_index], y.iloc[train_index])
                    acc_score = accuracy_score(y.iloc[val_index], model.predict(X.iloc[val_index]))
                    acc_scores.append(acc_score)
                    explainer = shap.TreeExplainer(model)
                    shap_value = explainer.shap_values(X, tree_limit=model.best_ntree_limit)
                    shap_values.append(shap_value)

                with open(self.output().pathlib_path / f"scores_{mtype}.txt", "a") as f:
                    print(np.mean(acc_scores), np.std(acc_scores), file=f)

                model.fit(X, y)
                model.save_model(self.output().pathlib_path / f"model_{mtype}.json")

                shap_val, _ = _get_shap_feature_importance(shap_values)

                shap.summary_plot(shap_val, X, plot_type="bar", max_display=20, show=False)
                plt.savefig(self.output().pathlib_path / f"bar_shap_{mtype}.pdf")
                plt.close()

                X.columns = [ALL_LABELS.get(col, col) for col in X.columns]
                shap.summary_plot(
                    shap_val,
                    X,
                    plot_type="dot",
                    max_display=10,
                    show=False,
                    color_bar_label="parameter value",
                    plot_size=(7, 4),
                )
                plt.tight_layout()
                plt.savefig(self.output().pathlib_path / f"dot_shap_{mtype}.pdf")
                plt.close()

    def output(self):
        """ """
        return EmodelLocalTarget(self.generalisation_models_path)


class CreateMLResistanceModels(WorkflowTask):
    """Create xgboost models of resistance parameter fits."""

    emodel = luigi.Parameter()

    resistance_models_path = luigi.Parameter(default="resistance_models")
    n_splits = luigi.IntParameter(default=10)
    n_repeats = luigi.IntParameter(default=5)

    def requires(self):
        """ """
        return {
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
            "resistance": ResistanceModels(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(exist_ok=True)
        resistance_models = yaml.safe_load(self.input()["resistance"].open())
        df_emodel = pd.read_csv(
            self.input()["emodel_parameters"].pathlib_path.parent / "emodels.csv",
            header=[0, 1],
            index_col=0,
        )
        df_emodel = df_emodel.set_index(("emodel", df_emodel["emodel"].columns[0]))

        param_bounds = {"ais": {}, "soma": {}}
        for tpe in ["ais", "soma"]:
            emodels = list(resistance_models[tpe].keys())
            X = df_emodel.loc[emodels, "normalized_parameters"]

            for val_id in range(4):
                y = []
                for emodel in emodels:
                    y.append(resistance_models[tpe][emodel]["resistance"]["polyfit_params"][val_id])
                params = []
                param_bounds[tpe][val_id] = {}
                for col in df_emodel["normalized_parameters"].columns:
                    x = df_emodel.loc[emodels, ("normalized_parameters", col)]
                    p = pearsonr(x, y)[0]
                    if abs(p) > 0.7:
                        param_bounds[tpe][val_id][col] = [
                            float(np.percentile(X[col], 10)),
                            float(np.percentile(X[col], 90)),
                        ]
                        params.append(col)
                        plt.figure(figsize=(5, 3))
                        plt.scatter(y, x)
                        plt.axhline(param_bounds[tpe][val_id][col][0])
                        plt.axhline(param_bounds[tpe][val_id][col][1])
                        plt.xlabel(f"{tpe} fit val: {val_id}")
                        plt.ylabel(ALL_LABELS.get(col, col))
                        plt.tight_layout()
                        plt.savefig(self.output().pathlib_path / f"corr_{tpe}_{val_id}_{col}.pdf")
                        plt.close()

                if not param_bounds[tpe][val_id]:
                    param_bounds[tpe][val_id] = float(np.mean(y))
                else:
                    _X = X[sorted(params)]

                    y = pd.Series(y)

                    model = XGBRegressor(learning_rate=0.1)
                    folds = RepeatedKFold(
                        n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=42
                    )
                    acc_scores = []
                    shap_values = []
                    for indices in tqdm(folds.split(_X, y=y), total=self.n_splits * self.n_repeats):
                        train_index, val_index = indices
                        model.fit(_X.iloc[train_index], y.iloc[train_index])
                        acc_score = mean_absolute_error(
                            y.iloc[val_index], model.predict(_X.iloc[val_index])
                        )
                        acc_scores.append(acc_score)
                        explainer = shap.TreeExplainer(model)
                        shap_value = explainer.shap_values(_X, tree_limit=model.best_ntree_limit)
                        shap_values.append(shap_value)

                    with open(self.output().pathlib_path / "scores.txt", "a") as f:
                        print(tpe, val_id, np.mean(acc_scores), np.std(acc_scores), file=f)

                    model.fit(_X, y)
                    model.save_model(self.output().pathlib_path / f"model_{tpe}_{val_id}.json")

                    shap_val, _ = _get_shap_feature_importance(shap_values)

                    _X.columns = [ALL_LABELS.get(col, col) for col in _X.columns]
                    shap.summary_plot(shap_val, _X, plot_type="bar", max_display=5, show=False)
                    plt.savefig(self.output().pathlib_path / f"bar_shap_{tpe}_{val_id}.pdf")
                    plt.close()

                    shap.summary_plot(shap_val, _X, plot_type="dot", max_display=5, show=False)
                    plt.savefig(self.output().pathlib_path / f"dot_shap_{tpe}_{val_id}.pdf")
                    plt.close()

        with open(self.output().pathlib_path / "param_bounds.yaml", "w") as f:
            yaml.dump(param_bounds, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.resistance_models_path)


class GenerateMoreModels(WorkflowTask):
    """Generate more models from MCMC with rho factor and resistances."""

    emodel = luigi.Parameter()
    n_models = luigi.IntParameter(default=500)
    more_models_path = luigi.Parameter(default="more_models")
    with_parameter_model = luigi.BoolParameter(default=True)
    with_bounds = luigi.BoolParameter(default=True)

    def requires(self):
        """ """
        tasks = {
            "mcmc": RunMCMC(emodel=self.emodel),
            "rho_models": CreateMLRhoModels(emodel=self.emodel),
            "resistance_models": CreateMLResistanceModels(emodel=self.emodel),
            "resistance": ResistanceModels(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }
        if self.with_parameter_model:
            tasks["generalisation_models"] = CreateMLGeneralisationModels(emodel=self.emodel)

        return tasks

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(exist_ok=True)

        mcmc_df = load_chains(self.input()["mcmc"].path)
        mcmc_df = mcmc_df[mcmc_df.cost < SelectRobustParams(emodel=self.emodel).split]

        for gid in mcmc_df.index:
            mcmc_df.loc[gid, "emodel"] = f"{self.emodel}_{gid}"
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())

        for mtype in exemplar_data["paths"].keys():
            print(mtype)
            if mtype != "all":
                target_rhos = {"rho": {}, "rho_axon": {}}
                if self.with_parameter_model:
                    model = XGBClassifier()
                    model.load_model(
                        self.input()["generalisation_models"].pathlib_path / f"model_{mtype}.json"
                    )
                    X = mcmc_df["normalized_parameters"]
                    prediction = model.predict(X)
                    _df = mcmc_df[prediction == 1]
                else:
                    _df = mcmc_df.copy()
                _df = _df.sample(min(len(_df), self.n_models), random_state=42)
                with open(
                    self.input()["rho_models"].pathlib_path / f"param_bounds_{mtype}.yaml"
                ) as f:
                    rho_param_bounds = yaml.safe_load(f)
                for rho in ["rho", "rho_axon"]:
                    if not isinstance(rho_param_bounds[rho], dict):
                        for emodel in _df["emodel"]:
                            target_rhos[rho][emodel] = rho_param_bounds[rho]
                    else:
                        if self.with_bounds:
                            for p, b in rho_param_bounds[rho].items():
                                _df = _df[
                                    (_df["normalized_parameters"][p] > b[0])
                                    & (_df["normalized_parameters"][p] < b[1])
                                ]

                        params = list(rho_param_bounds[rho].keys())

                        model_rho = XGBRegressor()
                        model_rho.load_model(
                            self.input()["rho_models"].pathlib_path / f"model_{mtype}_{rho}.json"
                        )
                        rhos = model_rho.predict(_df["normalized_parameters"][params])
                        for emodel, _rho in zip(_df["emodel"], rhos):
                            target_rhos[rho][emodel] = float(_rho)

                with open(self.output().pathlib_path / f"rho_factors_{mtype}.yaml", "w") as f:
                    yaml.dump(target_rhos, f)

                resistance_models = yaml.safe_load(self.input()["resistance"].open())

                shape = {
                    "soma": resistance_models["soma"][list(resistance_models["soma"].keys())[0]][
                        "shape"
                    ],
                    "ais": resistance_models["ais"][list(resistance_models["ais"].keys())[0]][
                        "shape"
                    ],
                }
                more_resistance_models = {"ais": {}, "soma": {}}
                with open(
                    self.input()["resistance_models"].pathlib_path / "param_bounds.yaml"
                ) as f:
                    param_bounds = yaml.safe_load(f)
                for tpe in ["ais", "soma"]:
                    for val_id in range(4):
                        if not isinstance(param_bounds[tpe][val_id], dict):
                            for emodel in _df["emodel"]:
                                if emodel not in more_resistance_models[tpe]:
                                    more_resistance_models[tpe][emodel] = {
                                        "shape": shape[tpe],
                                        "resistance": {"polyfit_params": 4 * [0]},
                                    }

                                more_resistance_models[tpe][emodel]["resistance"]["polyfit_params"][
                                    val_id
                                ] = float(param_bounds[tpe][val_id])
                        else:
                            model = XGBRegressor()
                            model.load_model(
                                self.input()["resistance_models"].pathlib_path
                                / f"model_{tpe}_{val_id}.json"
                            )
                            if self.with_bounds:
                                for p, b in param_bounds[tpe][val_id].items():
                                    _df = _df[
                                        (_df["normalized_parameters"][p] > b[0])
                                        & (_df["normalized_parameters"][p] < b[1])
                                    ]
                            params = sorted(list(param_bounds[tpe][val_id].keys()))
                            resistance = model.predict(_df["normalized_parameters"][params])
                            for emodel, res in zip(_df["emodel"], resistance):
                                if emodel not in more_resistance_models[tpe]:
                                    more_resistance_models[tpe][emodel] = {
                                        "shape": shape[tpe],
                                        "resistance": {"polyfit_params": 4 * [0]},
                                    }
                                more_resistance_models[tpe][emodel]["resistance"]["polyfit_params"][
                                    val_id
                                ] = float(res)

                with open(self.output().pathlib_path / f"resistance_model_{mtype}.yaml", "w") as f:
                    yaml.dump(more_resistance_models, f)

                save_selected_emodels(
                    _df,
                    _df.index,
                    emodel=self.emodel,
                    final_path=self.output().pathlib_path / f"final_{mtype}.json",
                )

    def output(self):
        """ """
        return EmodelLocalTarget(self.more_models_path)


class ValidateMoreModels(WorkflowTask):
    """Validation plot for generated models."""

    emodel = luigi.Parameter()
    more_models_validation = luigi.Parameter(default="more_models_validation")
    clip = luigi.FloatParameter(default=5)

    def requires(self):
        """ """
        return {
            "models": EvaluateMoreModels(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(exist_ok=True)
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                df = pd.read_csv(self.input()["models"].pathlib_path / f"evaluation_df_{mtype}.csv")

                score_df = get_score_df(df, FEATURE_FILTER)
                df.loc[:, "cost"] = score_df.max(1).to_list()

                df.loc[
                    score_df["SearchThresholdCurrent.soma.v.bpo_threshold_current"] > self.clip,
                    "cost",
                ] = 1000
                select_df = df.pivot(index="name", columns="emodel", values="cost")

                fail = select_df == 1000
                select_df[select_df <= self.clip] = 0
                select_df[select_df > self.clip] = 1
                select_df[select_df.isna()] = 1
                select_df[fail] = 2

                print(
                    mtype,
                    " good : ",
                    len(df[df.cost < self.clip]),
                    "/",
                    len(df),
                    " = ",
                    len(df[df.cost < self.clip]) / len(df),
                )

                print(
                    mtype,
                    " borderline: ",
                    len(df[df.cost < 900]),
                    "/",
                    len(df),
                    " = ",
                    len(df[df.cost < 900]) / len(df),
                )

                morph_select_df = select_df.sum(1).sort_values(ascending=False)
                emodel_select_df = select_df.sum(0).sort_values(ascending=False)
                select_df = select_df.loc[morph_select_df.index, emodel_select_df.index]

                plt.figure(figsize=(5, 4))
                plt.imshow(
                    select_df.to_numpy(),
                    aspect="auto",
                    origin="lower",
                    cmap="Greys",
                    interpolation="nearest",
                )
                plt.colorbar(shrink=0.8, label="cost")
                plt.ylabel("synthesized morphologies")
                plt.xlabel("electrical models")
                plt.tight_layout()
                plt.savefig(self.output().pathlib_path / f"cost_heatmap_{mtype}.pdf")

    def output(self):
        """ """
        return EmodelLocalTarget(self.more_models_validation)


class EvaluateMoreModels(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Generate more models from MCMC with rho factor and resistances."""

    emodel = luigi.Parameter()
    n_steps = luigi.IntParameter(default=2)
    with_soma = luigi.BoolParameter(default=True)
    min_scale = luigi.FloatParameter(default=0.01)
    max_scale = luigi.FloatParameter(default=10.0)

    more_models_evaluation_df = luigi.Parameter(default="more_models_evaluation")

    def requires(self):
        """ """
        return {
            "models": GenerateMoreModels(emodel=self.emodel),
            "selected": SelectValidParameters(emodel=self.emodel),
            "combodb": ReDiametrize(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        self.output().pathlib_path.mkdir(exist_ok=True)
        trace_folder = self.output().pathlib_path / "trace_data"
        trace_folder.mkdir(exist_ok=True)

        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                with open(
                    self.input()["models"].pathlib_path / f"resistance_model_{mtype}.yaml", "r"
                ) as f:
                    resistance_models = yaml.safe_load(f)
                with open(self.input()["selected"].pathlib_path / f"selected_{mtype}.yaml") as f:
                    selected = yaml.safe_load(f)
                with open(
                    self.input()["models"].pathlib_path / f"rho_factors_{mtype}.yaml", "r"
                ) as f:
                    rhos = yaml.safe_load(f)

                combos_df = pd.read_csv(self.input()["combodb"].path)

                combos_df["ais_model"] = json.dumps(exemplar_data["ais"])
                combos_df["ais_scaler"] = 1.0
                combos_df["soma_model"] = json.dumps(exemplar_data["soma"])
                combos_df["soma_scaler"] = 1.0

                final_path = str(self.input()["models"].pathlib_path / f"final_{mtype}.json")
                morphs = selected["morphos"]
                with open(final_path) as f:
                    emodels = list(json.load(f).keys())
                pairs = list(itertools.product(morphs, emodels))
                _morphs = [p[0] for p in pairs]
                _emodels = [p[1] for p in pairs]
                df = combos_df.set_index("name").loc[_morphs].reset_index()
                df["emodel"] = _emodels
                df = adapt_soma_ais(
                    df,
                    access_point=self.get_access_point(final_path=final_path),
                    models=resistance_models,
                    rhos=rhos,
                    parallel_factory=self.parallel_factory,
                    n_steps=self.n_steps,
                    min_scale=self.min_scale,
                    max_scale=self.max_scale,
                )

                df = evaluate_rho_axon(
                    df,
                    self.get_access_point(final_path=final_path),
                    parallel_factory=self.parallel_factory,
                )

                df = evaluate_rho(
                    df,
                    self.get_access_point(final_path=final_path),
                    parallel_factory=self.parallel_factory,
                )

                df = feature_evaluation(
                    df,
                    self.get_access_point(final_path=final_path),
                    parallel_factory=self.parallel_factory,
                    trace_data_path=trace_folder,
                )
                df.to_csv(self.output().pathlib_path / f"evaluation_df_{mtype}.csv", index=False)

    def output(self):
        """ """
        return EmodelLocalTarget(self.more_models_evaluation_df)


class Evaluate(EmodelAwareTask, ParallelTask, WorkflowTask):
    """Test the various parameters using AIS/soma adaptation."""

    emodel = luigi.Parameter()
    evaluation_path = luigi.Parameter(default="evaluations.csv")
    with_traces = luigi.BoolParameter(default=True)
    with_adaptation = luigi.BoolParameter(default=True)

    def requires(self):
        """ """
        tasks = {
            "emodel_config": GetEmodelConfig(emodel=self.emodel),
            "emodel_parameters": SelectRobustParams(emodel=self.emodel),
        }
        if self.with_adaptation:
            tasks["adapt"] = AdaptAisSoma(emodel=self.emodel)
        else:
            tasks["combodb"] = ReDiametrize(emodel=self.emodel)
            tasks["exemplar"] = CreateExemplar(emodel=self.emodel)
        return tasks

    def run(self):
        """ """
        if self.with_adaptation:
            df = pd.read_csv(self.input()["adapt"].path)
        else:
            exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
            with open(self.input()["emodel_parameters"].path) as f:
                emodels = list(json.load(f).keys())
            combos_df = pd.read_csv(self.input()["combodb"].path)
            df = make_evaluation_df(combos_df, emodels, exemplar_data)

        final_path = self.input()["emodel_parameters"].path
        trace_folder = None
        if self.with_traces:
            trace_folder = self.output().pathlib_path.parent / "trace_data"
            trace_folder.mkdir(exist_ok=True)

        df = feature_evaluation(
            df,
            self.get_access_point(final_path=final_path),
            parallel_factory=self.parallel_factory,
            trace_data_path=trace_folder,
        )
        df.to_csv(self.output().path, index=False)

    def output(self):
        """ """
        if self.with_adaptation:
            return EmodelLocalTarget(self.evaluation_path)
        return EmodelLocalTarget(
            str(Path(self.evaluation_path).with_suffix("")) + "_no_adaptation.csv"
        )


class PlotTraces(WorkflowTask):
    """Plot traces, higlhlighting exemplar."""

    emodel = luigi.Parameter()
    trace_plots = luigi.Parameter(default="trace_plots")

    def requires(self):
        """ """
        return {
            "exemplar": CreateExemplar(emodel=self.emodel),
            "evaluate": Evaluate(emodel=self.emodel),
            "select": SelectValidParameters(emodel=self.emodel),
        }

    def run(self):
        """ """
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        df = pd.read_csv(self.input()["evaluate"].path)

        self.output().pathlib_path.mkdir(exist_ok=True)
        trace_folder = self.input()["evaluate"].pathlib_path.parent / "trace_data"

        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                print(f"Plotting {mtype}")
                with open(self.input()["select"].pathlib_path / f"selected_{mtype}.yaml") as f:
                    select = yaml.safe_load(f)

                _df = df[df.emodel.isin(select["emodels"])]
                _df = _df[df.name.isin(select["morphos"])]

                _df["trace_highlight"] = False
                _df.loc[df.path == exemplar_data["paths"][mtype], "trace_highlight"] = True
                for emodel in tqdm(_df.emodel.unique()):
                    plot_traces(
                        _df[_df.emodel == emodel],
                        trace_path=trace_folder,
                        pdf_filename=self.output().pathlib_path
                        / f"trace_plot_{mtype}_{emodel}.pdf",
                    )

    def output(self):
        """ """
        return EmodelLocalTarget(self.trace_plots)


class SelectMoreValidParameters(EmodelAwareTask, WorkflowTask):
    """Select valid set of pameters."""

    emodel = luigi.Parameter()
    valid_emodels_path = luigi.Parameter(default="selected_more")
    clip = luigi.FloatParameter(default=5)
    morpho_thresh = luigi.FloatParameter(default=0.2)
    emodel_thresh = luigi.FloatParameter(default=0.05)

    def requires(self):
        """ """
        return {
            "evaluate": EvaluateMoreModels(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())
        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                df = pd.read_csv(
                    self.input()["evaluate"].pathlib_path / f"evaluation_df_{mtype}.csv"
                )
                selected = select_valid(
                    df[df.mtype == mtype],
                    self.emodel,
                    self.output().pathlib_path,
                    self.morpho_thresh,
                    self.emodel_thresh,
                    self.clip,
                    self.get_access_point(),
                    mtype,
                )

                with open(self.output().pathlib_path / f"selected_{mtype}.yaml", "w") as f:
                    yaml.dump(selected, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.valid_emodels_path)


class SelectValidParameters(EmodelAwareTask, WorkflowTask):
    """Select valid set of pameters."""

    emodel = luigi.Parameter()
    valid_emodels_path = luigi.Parameter(default="selected")
    clip = luigi.FloatParameter(default=5)
    morpho_thresh = luigi.FloatParameter(default=0.2)
    emodel_thresh = luigi.FloatParameter(default=0.05)

    def requires(self):
        """ """
        return {
            "evaluate": Evaluate(emodel=self.emodel),
        }

    def run(self):
        """ """
        df = pd.read_csv(self.input()["evaluate"].path)
        self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

        for mtype in df.mtype.unique():
            selected = select_valid(
                df[df.mtype == mtype],
                self.emodel,
                self.output().pathlib_path,
                self.morpho_thresh,
                self.emodel_thresh,
                self.clip,
                self.get_access_point(),
                mtype,
            )

            with open(self.output().pathlib_path / f"selected_{mtype}.yaml", "w") as f:
                yaml.dump(selected, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.valid_emodels_path)


class SelectValidParametersNoAdapt(EmodelAwareTask, WorkflowTask):
    """Select valid set of pameters."""

    emodel = luigi.Parameter()
    valid_emodels_path = luigi.Parameter(default="selected_no_adapt")
    clip = luigi.FloatParameter(default=5)
    morpho_thresh = luigi.FloatParameter(default=0.2)
    emodel_thresh = luigi.FloatParameter(default=0.05)

    def requires(self):
        """ """
        return {
            "evaluate": Evaluate(emodel=self.emodel, with_adaptation=False),
            "evaluate_adapted": Evaluate(emodel=self.emodel),
        }

    def run(self):
        """ """
        df = pd.read_csv(self.input()["evaluate"].path)
        df_adapted = pd.read_csv(self.input()["evaluate_adapted"].path)
        df = df[df.emodel.isin(df_adapted.emodel.unique())]
        self.output().pathlib_path.mkdir(parents=True, exist_ok=True)

        for mtype in df.mtype.unique():
            selected = select_valid(
                df[df.mtype == mtype],
                self.emodel,
                self.output().pathlib_path,
                self.morpho_thresh,
                self.emodel_thresh,
                self.clip,
                self.get_access_point(),
                mtype,
            )
            with open(self.output().pathlib_path / f"selected_no_adapt_{mtype}.yaml", "w") as f:
                yaml.dump(selected, f)

    def output(self):
        """ """
        return EmodelLocalTarget(self.valid_emodels_path)


class ValidateMorphologiesNoAdapt(WorkflowTask):
    """Asses which morphologies are outliers."""

    emodel = luigi.Parameter()
    validation_path = luigi.Parameter(default="validation_morphology_no_adapt")

    def requires(self):
        """ """
        return {
            "evaluate": Evaluate(emodel=self.emodel, with_adaptation=False),
            "selected": SelectValidParametersNoAdapt(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(parents=True)
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())

        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                with open(
                    self.input()["selected"].pathlib_path / f"selected_no_adapt_{mtype}.yaml"
                ) as f:
                    selected = yaml.safe_load(f)
                df = pd.read_csv(self.input()["evaluate"].path)
                validate_morphologies(
                    df,
                    selected,
                    self.output().pathlib_path,
                    self.input()["selected"].pathlib_path,
                    exemplar_data,
                    mtype,
                )

    def output(self):
        """ """
        return EmodelLocalTarget(self.validation_path)


class ValidateMoreMorphologies(WorkflowTask):
    """Asses which morphologies are outliers."""

    emodel = luigi.Parameter()
    validation_path = luigi.Parameter(default="validation_more_morphology")

    def requires(self):
        """ """
        return {
            "evaluate": EvaluateMoreModels(emodel=self.emodel),
            "selected": SelectValidParameters(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(parents=True)
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())

        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                with open(self.input()["selected"].pathlib_path / f"selected_{mtype}.yaml") as f:
                    selected = yaml.safe_load(f)
                df = pd.read_csv(self.input()["evaluate"].path)
                validate_morphologies(
                    df,
                    selected,
                    self.output().pathlib_path,
                    self.input()["selected"].pathlib_path,
                    exemplar_data,
                    mtype,
                )

    def output(self):
        """ """
        return EmodelLocalTarget(self.validation_path)


class ValidateMorphologies(WorkflowTask):
    """Asses which morphologies are outliers."""

    emodel = luigi.Parameter()
    validation_path = luigi.Parameter(default="validation_morphology")

    def requires(self):
        """ """
        return {
            "evaluate": Evaluate(emodel=self.emodel),
            "selected": SelectValidParameters(emodel=self.emodel),
            "exemplar": CreateExemplar(emodel=self.emodel),
        }

    def run(self):
        """ """
        self.output().pathlib_path.mkdir(parents=True)
        exemplar_data = yaml.safe_load(self.input()["exemplar"].open())

        for mtype in exemplar_data["paths"].keys():
            if mtype != "all":
                with open(self.input()["selected"].pathlib_path / f"selected_{mtype}.yaml") as f:
                    selected = yaml.safe_load(f)
                df = pd.read_csv(self.input()["evaluate"].path)
                validate_morphologies(
                    df,
                    selected,
                    self.output().pathlib_path,
                    self.input()["selected"].pathlib_path,
                    exemplar_data,
                    mtype,
                )

    def output(self):
        """ """
        return EmodelLocalTarget(self.validation_path)


class CreateSynthesisComboDF(WorkflowTask):
    """Create dataframe with combos mixing mtype, etype and emodels for synthesized cells."""

    emodel = luigi.Parameter()


class CompareNoAdaptation(WorkflowTask):
    """Compare non adapted cells."""

    emodel = luigi.Parameter()
    adaptation_comp_figure = luigi.Parameter(default="adaptation_comparison.pdf")

    def requires(self):
        """ """
        return {
            "adapted": Evaluate(emodel=self.emodel),
            "not_adapted": Evaluate(emodel=self.emodel, with_adaptation=False),
        }

    def run(self):
        """ """
        clip = 5
        df_adapted = pd.read_csv(self.input()["adapted"].path)
        df_not_adapted = pd.read_csv(self.input()["not_adapted"].path)
        plot_adaptation_summaries(
            df_adapted,
            df_not_adapted,
            filename=self.output().pathlib_path.parent / "adaptation_summary.pdf",
        )
        compare_adaptation(df_adapted, df_not_adapted, clip, self.output().path)

    def output(self):
        """ """
        return EmodelLocalTarget(self.adaptation_comp_figure)


class Run(WorkflowWrapperTask):
    """Main task to run the entire workflow."""

    emodels = luigi.ListParameter(default=["cADpyr_L5"])
    output_folder = luigi.Parameter(default="out")
    with_generalisation = luigi.BoolParameter(default=False)
    with_information_theory = luigi.BoolParameter(default=False)
    with_no_adaptation = luigi.BoolParameter(default=False)

    def requires(self):
        """ """
        for emodel in self.emodels:  # pylint: disable=not-an-iterable
            EmodelLocalTarget.set_default_prefix(emodel)
            tasks = [
                GetEmodelConfig(emodel=emodel),
                CreateComboDF(emodel=emodel),
                ReDiametrize(emodel=emodel),
                CreateExemplar(emodel=emodel),
                RunMCMC(emodel=emodel),
                PlotMCMCResults(emodel=emodel),
            ]
            if self.with_generalisation:
                tasks.append(SelectRobustParams(emodel=emodel))
                tasks += [
                    RhoFactors(emodel=emodel),
                    CreateMLRhoModels(emodel=emodel),
                    AdaptAisSoma(emodel=emodel),
                    Evaluate(emodel=emodel),
                    ValidateMorphologies(emodel=emodel),
                    SelectValidParameters(emodel=emodel),
                    CreateMLResistanceModels(emodel=emodel),
                    GenerateMoreModels(emodel=emodel),
                    EvaluateMoreModels(emodel=emodel),
                    ValidateMoreModels(emodel=emodel),
                    SelectMoreValidParameters(emodel=emodel),
                    PlotTraces(emodel=emodel),
                    # BAPValidation(emodel=emodel),
                    # ValidateParamsOnSynthesis(emodel=emodel),
                ]
                if self.with_no_adaptation:
                    tasks.append(ValidateMorphologiesNoAdapt(emodel=emodel))
                    tasks.append(CompareNoAdaptation(emodel=emodel))
                    tasks.append(SelectValidParametersNoAdapt(emodel=emodel))
        return tasks


OutputLocalTarget.set_default_prefix(Run().output_folder)
