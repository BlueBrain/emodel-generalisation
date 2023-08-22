"""Cli app."""
import json
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from datareuse import Reuse
from matplotlib.backends.backend_pdf import PdfPages
from morphio import Morphology
from tqdm import tqdm
from voxcell import CellCollection

from emodel_generalisation.adaptation import adapt_soma_ais
from emodel_generalisation.adaptation import build_all_resistance_models
from emodel_generalisation.bluecellulab_evaluator import evaluate_currents
from emodel_generalisation.exemplars import extract_ais_diameters
from emodel_generalisation.exemplars import extract_ais_path_distances
from emodel_generalisation.exemplars import generate_exemplars
from emodel_generalisation.mcmc import plot_feature_distributions
from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.model.evaluation import create_cell_model
from emodel_generalisation.model.evaluation import evaluate_rho
from emodel_generalisation.model.evaluation import evaluate_rho_axon
from emodel_generalisation.model.evaluation import feature_evaluation
from emodel_generalisation.model.modifiers import get_replace_axon_hoc
from emodel_generalisation.parallal import init_parallel_factory
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_feature_df
from emodel_generalisation.utils import get_score_df

L = logging.getLogger(__name__)
_BASE_PATH = Path(__file__).parent.resolve()

# pylint: disable=too-many-locals


def _load_circuit(input_path, morphology_path):
    input_cells = CellCollection.load(input_path)
    cells_df = input_cells.as_dataframe()
    for column in cells_df.columns:
        if cells_df[column].dtype == "category":
            cells_df[column] = cells_df[column].astype(object)
    cells_df["emodel"] = cells_df["model_template"].apply(lambda temp: temp[4:])
    cells_df["path"] = [f"{morphology_path}/{m}.asc" for m in cells_df["morphology"]]
    return cells_df, input_cells


@click.group()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    """Cli."""
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=loglevel, format=logformat)


@cli.command("compute_currents")
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", default="circuit_currents.h5", type=str)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--hoc-path", type=str, required=True)
@click.option("--protocol-config-path", type=str, required=True)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--resume", is_flag=True)
@click.option("--sql-tmp-path", default=None, type=str)
@click.option("--debug-csv-path", default=None, type=str)
def compute_currents(
    input_path,
    output_path,
    morphology_path,
    hoc_path,
    protocol_config_path,
    parallel_lib,
    resume,
    sql_tmp_path,
    debug_csv_path,
):
    """For each cell, compute the holding, thresholds currents as well as Rin/RMP.

    Args:
        output_path (str): path to sonata file to save output data for all me-combos
        output_path (str): path to save new sonata file
        morphology_path (str): base path to morphologies, if none, same dir as mvd3 will be used
        hoc_path (str): base path to hoc files
        protocol_config_path (str): path to yaml file with protocol config
        parallel_lib (str): parallel library
        resume (bool): resume computation
        sql_tmp_path (str): path to a folder to save sql files used during computations
        debug_csv_path (str): to save a debug file with more info than sonata (exceptions, etc...)
    """
    if resume and sql_tmp_path is not None:
        raise Exception("If --sql-tmp-path is not set, --resume cannot work")

    parallel_factory = init_parallel_factory(parallel_lib)
    cells_df, input_cells = _load_circuit(input_path, morphology_path)

    with open(protocol_config_path, "r") as prot_file:
        protocol_config = yaml.safe_load(prot_file)

    cells_df = evaluate_currents(
        cells_df,
        protocol_config,
        hoc_path,
        parallel_factory=parallel_factory,
        db_url=Path(sql_tmp_path) / "current_db.sql" if sql_tmp_path is not None else None,
        template_format="v6_adapted" if "@dynamics:AIS_scaler" in cells_df.columns else "v6",
        resume=resume,
    )
    if debug_csv_path is not None:
        cells_df.to_csv(debug_csv_path, index=False)
        for col in [
            "@dynamics:holding_current",
            "@dynamics:threshold_current",
            "@dynamics:resting_potential",
            "@dynamics:input_resistance",
        ]:
            if col in cells_df.columns:
                cells_df = cells_df.drop(col, axis=1)
    cells_df = cells_df.rename(
        columns={
            "holding_current": "@dynamics:holding_current",
            "threshold_current": "@dynamics:threshold_current",
            "resting_potential": "@dynamics:resting_potential",
            "input_resistance": "@dynamics:input_resistance",
        }
    ).drop(columns=["path", "exception", "emodel"])

    failed_cells = cells_df[cells_df.isnull().any(axis=1)].index
    if len(failed_cells) > 0:
        L.info("%s failed cells:", len(failed_cells))
        L.info(cells_df.loc[failed_cells])
        cells_df.loc[failed_cells, "mtype"] = None

    output_cells = CellCollection.from_dataframe(cells_df)
    output_cells.population_name = input_cells.population_name
    output_cells.remove_unassigned_cells()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_cells.save(output_path)

    # clean up the parallel library, if needed
    parallel_factory.shutdown()


def plot_evaluation(cells_df, access_point, main_path="analysis_plot", clip=5, feature_filter=None):
    """Make some plots of evaluations."""
    main_path = Path(main_path)
    main_path.mkdir(exist_ok=True)
    if feature_filter is None or feature_filter == "":
        feature_filter = FEATURE_FILTER
        feature_filter.append("inv_time_to_first_spike")
        feature_filter.append("burst_number")
        feature_filter.append("ohmic_input_resistance_vb_ssse")
        feature_filter.append("AHP_depth_abs")
        feature_filter.append("AHP_depth")
    else:
        feature_filter = json.loads(feature_filter)

    L.info("Plotting summary figure...")
    scores = get_score_df(cells_df, filters=feature_filter)
    cells_df["cost"] = np.clip(scores.max(1), 0, clip)
    _df = cells_df[["emodel", "mtype", "cost"]].groupby(["emodel", "mtype"]).mean().reset_index()
    plot_df = _df.pivot(index="emodel", columns="mtype", values="cost")
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        plot_df,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=clip,
        ax=plt.gca(),
        cbar_kws={"label": f"mean cost (clip {clip})", "shrink": 0.5},
    )
    plt.tight_layout()
    plt.savefig(main_path / "cost.pdf")
    plt.close()

    L.info("Plotting worst scores...")
    with PdfPages(main_path / "worst_scores.pdf") as _pdf:
        for emodel, emodel_df in cells_df.groupby("emodel"):
            _df = get_score_df(emodel_df, filters=feature_filter)
            max_feat = pd.DataFrame(columns=_df.columns)
            for mtype, mtype_df in emodel_df.groupby("mtype"):
                _df = get_score_df(mtype_df, filters=feature_filter).T.clip(0, clip)
                m = _df.max(axis=0)
                _df["count"] = (_df == m).sum(axis=1)
                max_feat.loc[mtype] = _df["count"]
            max_feat[max_feat == 0] = np.nan
            max_feat = max_feat.dropna(axis=1, how="all")
            # pylint: disable=unsupported-assignment-operation
            max_feat["all"] = max_feat.sum(axis=1)
            max_feat = max_feat.sort_values(by="all", ascending=True).drop(columns=["all"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                max_feat.T,
                xticklabels=True,
                yticklabels=True,
                ax=plt.gca(),
                cbar_kws={"label": "# blocker feature", "shrink": 0.5},
            )
            plt.suptitle(f"emodel={emodel}")
            plt.tight_layout()
            _pdf.savefig()
            plt.close()

    L.info("Plotting feature distributions...")
    path = main_path / "feature_distributions"
    path.mkdir(exist_ok=True)
    for (emodel, mtype), df in tqdm(cells_df.groupby(["emodel", "mtype"])):
        _df = {"features": get_feature_df(df, filters=feature_filter)}
        try:
            plot_feature_distributions(
                _df,
                emodel,
                access_point,
                path / f"distribution_{emodel}_{mtype}.pdf",
                log_scale=False,
            )
        except ValueError:
            L.warning("Cannot plot distributions of %s %s", emodel, mtype)


@cli.command("evaluate")
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", default="evaluation_df.csv", type=str)
@click.option("--n-cells-per-emodel", default=10, type=int)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--config-path", type=str, required=True)
@click.option("--final-path", type=str, required=True)
@click.option("--legacy", is_flag=True)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--resume", is_flag=True)
@click.option("--sql-tmp-path", default=None, type=str)
@click.option("--clip", default=5, type=float)
@click.option("--feature-filter", default="", type=str)
@click.option("--validation-path", default="analysis_plot", type=str)
@click.option("--with-model-management", is_flag=True)
def evaluate(
    input_path,
    output_path,
    n_cells_per_emodel,
    morphology_path,
    config_path,
    final_path,
    legacy,
    parallel_lib,
    resume,
    sql_tmp_path,
    clip,
    feature_filter,
    validation_path,
    with_model_management,
):
    """Evaluate models from a circuit."""
    parallel_factory = init_parallel_factory(parallel_lib)
    if legacy:
        access_point = AccessPoint(
            emodel_dir=Path(config_path),
            final_path=final_path,
            legacy_dir_structure=True,
            with_seeds=True,
        )
    else:
        access_point = AccessPoint(
            emodel_dir=Path(config_path).parent,
            final_path=final_path,
            recipes_path=Path(config_path) / "recipes.json",
            with_seeds=True,
        )

    cells_df, _ = _load_circuit(input_path, morphology_path)
    cells_df = (
        cells_df.groupby(["emodel", "mtype"])
        .sample(n_cells_per_emodel, random_state=42, replace=True)
        .reset_index()
    )

    # add data for adapted AIS/soma if available
    if Path("exemplar_data.yaml").exists():
        with open("exemplar_data.yaml") as exemplar_f:
            exemplar_data = yaml.safe_load(exemplar_f)

        for emodel, data in exemplar_data.items():
            cells_df.loc[cells_df.emodel == emodel, "ais_model"] = json.dumps(data["ais"])
            cells_df.loc[cells_df.emodel == emodel, "soma_model"] = json.dumps(data["soma"])
            cells_df = cells_df.rename(
                columns={
                    "@dynamics:AIS_scaler": "ais_scaler",
                    "@dynamics:soma_scaler": "soma_scaler",
                }
            )

    with Reuse(output_path, index=False) as reuse:
        cells_df = reuse(
            feature_evaluation,
            cells_df,
            access_point,
            parallel_factory=parallel_factory,
            resume=resume,
            db_url=sql_tmp_path,
        )

    if with_model_management:
        exemplar_df = pd.DataFrame()
        for gid, emodel in enumerate(cells_df.emodel.unique()):
            morph = access_point.get_morphologies(emodel)
            exemplar_df.loc[gid, "emodel"] = emodel
            exemplar_df.loc[gid, "path"] = morph["path"]
            exemplar_df.loc[gid, "name"] = morph["name"]

        with Reuse(str(Path(output_path).with_suffix("")) + "_exemplar.csv") as reuse:
            exemplar_df = reuse(
                feature_evaluation,
                exemplar_df,
                access_point,
                parallel_factory=parallel_factory,
                resume=resume,
                db_url=sql_tmp_path,
            )
        cells_score_df = get_score_df(cells_df, filters=feature_filter)
        exemplar_df = exemplar_df.set_index("emodel").loc[cells_df.emodel].reset_index()
        exemplar_score_df = get_score_df(exemplar_df, filters=feature_filter)

        _pass = cells_score_df.copy()
        for col in cells_score_df.columns:
            _pass[col] = cells_score_df[col] <= np.maximum(5.0, 5.0 * exemplar_score_df[col])

        pass_df = pd.DataFrame()
        pass_df["pass"] = _pass.all(axis=1)
        pass_df["emodel"] = cells_df.emodel
        pass_df["mtype"] = cells_df.mtype
        mm = pass_df.groupby(["emodel", "mtype"]).mean()
        L.info("Result of model-management: %s", mm)
        mm.to_csv("model_management.csv")

    if validation_path:
        plot_evaluation(
            cells_df,
            access_point,
            main_path=validation_path,
            clip=clip,
            feature_filter=feature_filter,
        )


@cli.command("adapt")
@click.option("--input-node-path", type=click.Path(exists=True), required=True)
@click.option("--output-csv-path", default="adapt_df.csv", type=str)
@click.option("--output-node-path", default="node.h5", type=str)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--config-path", type=str, required=True)
@click.option("--final-path", type=str, required=True)
@click.option("--hoc-path", type=str, default="hoc")
@click.option("--template-path", type=str, default=None)
@click.option("--legacy", is_flag=True)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--resume", is_flag=True)
@click.option("--sql-tmp-path", default=None, type=str)
def adapt(
    input_node_path,
    output_csv_path,
    output_node_path,
    morphology_path,
    config_path,
    final_path,
    hoc_path,
    template_path,
    legacy,
    parallel_lib,
    resume,
    sql_tmp_path,
):
    """Adapt cells from a circuit with rho factors.

    This is an adapted version of the complete workflow, to work on an existing circuit.

    Here is how it works:
        1. extract AIS profile and soma surface area from exemplar morphologies
        2. compute their rho factors, to be used as references
        3. fit resistance models of AIS and soma
        4. adapt AIS/soma of all cells to match rho factors
        5. save circuit with @dynamics:AIS_scaler and @dynamics:soma_scaler entries
        6. create hoc files with corresponding replace_axon
    """
    parallel_factory = init_parallel_factory(parallel_lib)
    if legacy:
        access_point = AccessPoint(
            emodel_dir=Path(config_path),
            final_path=final_path,
            legacy_dir_structure=True,
            with_seeds=True,
        )
    else:
        access_point = AccessPoint(
            emodel_dir=Path(config_path).parent,
            final_path=final_path,
            recipes_path=Path(config_path) / "recipes.json",
            with_seeds=True,
        )

    cells_df, _ = _load_circuit(input_node_path, morphology_path)

    exemplar_df = pd.DataFrame()
    for gid, emodel in enumerate(cells_df.emodel.unique()):
        morph = access_point.get_morphologies(emodel)
        exemplar_df.loc[gid, "emodel"] = emodel
        exemplar_df.loc[gid, "path"] = morph["path"]
        exemplar_df.loc[gid, "name"] = morph["name"]

    # get exemplar data
    L.info("Extracting exemplar data...")

    def _get_ais_profile(path):
        """Create ais profile from exemplar, instead of constant mean diameters as by default.

        This is important to remain closer to the exemplar morphology used for optimisation.
        """
        morphology = Morphology(path)
        orig_lens = next(extract_ais_path_distances([morphology]))
        orig_diams = next(extract_ais_diameters([morphology]))
        lens = np.linspace(0, 60, 10)
        diams = np.interp(lens, orig_lens, orig_diams)
        return diams.tolist()

    def _get_exemplar_data(exemplar_df):
        """Create exemplar data for all emodels."""
        exemplar_data = {}
        for emodel in exemplar_df.emodel:
            _df = exemplar_df[exemplar_df.emodel == emodel]
            _df["mtype"] = "all"
            exemplar_data[emodel] = generate_exemplars(_df, with_plots=False, surface_percentile=50)
            exemplar_data[emodel]["ais"]["popt"] = _get_ais_profile(_df["path"].tolist()[0])

        return exemplar_data

    with Reuse("exemplar_data.yaml") as reuse:
        exemplar_data = reuse(_get_exemplar_data, exemplar_df)

    L.info("Compute exemplar rho factors...")
    for gid, emodel in enumerate(cells_df.emodel.unique()):
        exemplar_df.loc[gid, "ais_model"] = json.dumps(exemplar_data[emodel]["ais"])
        exemplar_df.loc[gid, "soma_model"] = json.dumps(exemplar_data[emodel]["soma"])
        exemplar_df.loc[gid, "soma_scaler"] = 1.0
        exemplar_df.loc[gid, "ais_scaler"] = 1.0

    with Reuse(str(Path(output_csv_path).with_suffix("")) + "_exemplar_rho.csv") as reuse:
        exemplar_df = reuse(
            evaluate_rho,
            exemplar_df,
            access_point,
            parallel_factory=parallel_factory,
            resume=resume,
            db_url=sql_tmp_path,
        )

    with Reuse(str(Path(output_csv_path).with_suffix("")) + "_exemplar_rho_axon.csv") as reuse:
        exemplar_df = reuse(
            evaluate_rho_axon,
            exemplar_df,
            access_point,
            parallel_factory=parallel_factory,
            resume=resume,
            db_url=sql_tmp_path,
        )

    L.info("Create resistance models of AIS and soma...")
    scales_params = {"min": -0.5, "max": 0.5, "n": 10, "lin": False}  # possibly configurable

    def _get_resistance_models(exemplar_df, exemplar_data, scales_params):
        """We fit the scale/Rin relation for AIS and soma."""
        models = {}
        for emodel in exemplar_df.emodel:
            models[emodel] = build_all_resistance_models(
                access_point, [emodel], exemplar_data[emodel], scales_params
            )
        return models

    with Reuse("resistance_models.yaml") as reuse:
        resistance_models = reuse(_get_resistance_models, exemplar_df, exemplar_data, scales_params)

    L.info("Adapting AIS and soma of all cells..")
    cells_df["ais_scaler"] = 1.0
    cells_df["soma_scaler"] = 1.0

    def _adapt():
        """Adapt AIS/somam scales to match the rho factors."""
        for i, emodel in enumerate(exemplar_df.emodel):
            L.info("Adapting model: %s, %s / %s", emodel, i, len(exemplar_df))
            rhos = (
                exemplar_df[exemplar_df.emodel == emodel][["rho", "rho_axon", "emodel"]]
                .set_index("emodel")
                .to_dict()
            )
            mask = cells_df.emodel == emodel

            cells_df.loc[mask, "ais_model"] = json.dumps(exemplar_data[emodel]["ais"])
            cells_df.loc[mask, "soma_model"] = json.dumps(exemplar_data[emodel]["soma"])
            cells_df[mask] = adapt_soma_ais(
                cells_df[mask],
                access_point,
                resistance_models[emodel],
                rhos,
                parallel_factory=parallel_factory,
                min_scale=0.5,
                max_scale=2.0,
                n_steps=2,
            )
        return cells_df

    with Reuse(output_csv_path) as reuse:
        cells_df = reuse(_adapt)

    # finally save a node.h5 file
    L.info("Saving sonata file...")
    cells = CellCollection.load(input_node_path)
    cells.properties["@dynamics:AIS_scaler"] = cells_df["ais_scaler"].to_list()
    cells.properties["@dynamics:soma_scaler"] = cells_df["soma_scaler"].to_list()
    cells.save(output_node_path)

    L.info("Create hoc files...")
    if template_path is None:
        template_path = _BASE_PATH / "templates" / "cell_template_neurodamus.jinja2"
    else:
        template_path = Path(template_path)

    for emodel in exemplar_df.emodel:
        hoc_params = [
            exemplar_data[emodel]["soma"]["soma_radius"],
            exemplar_data[emodel]["soma"]["soma_surface"],
        ] + exemplar_data[emodel]["ais"]["popt"]
        morph_modifier_hoc = get_replace_axon_hoc(hoc_params)
        model_configuration = access_point.get_configuration(emodel)
        cell_model = create_cell_model(
            emodel,
            model_configuration=model_configuration,
            morph_modifiers=[lambda: None],
            morph_modifiers_hoc=[morph_modifier_hoc],
        )
        hoc = cell_model.create_hoc(
            access_point.final[emodel]["params"],
            template=template_path.name,
            template_dir=template_path.parent,
        )
        Path(hoc_path).mkdir(exist_ok=True)
        with open(Path(hoc_path) / f"{emodel}.hoc", "w") as hoc_file:
            hoc_file.write(hoc)