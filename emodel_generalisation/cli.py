"""Cli app."""
import json
import logging
import os
import subprocess
from collections import defaultdict
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
from emodel_generalisation.parallel import init_parallel_factory
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_feature_df
from emodel_generalisation.utils import get_score_df

L = logging.getLogger(__name__)
_BASE_PATH = Path(__file__).parent.resolve()

# pylint: disable=too-many-locals


def _load_circuit(input_path, morphology_path=None, population_name=None):
    """Load a circuit into a dataframe."""
    input_cells = CellCollection.load_sonata(input_path, population_name)
    cells_df = input_cells.as_dataframe()

    if "model_template" in cells_df:
        cells_df["emodel"] = (
            cells_df["model_template"].apply(lambda temp: temp[4:]).astype("category")
        )

    if morphology_path is not None:
        cells_df["path"] = [f"{morphology_path}/{m}.asc" for m in cells_df["morphology"]]

    return cells_df, input_cells


def _get_access_point(config_path, final_path=None, legacy=False, local_config="config"):
    """Get access point."""
    config_path = Path(config_path)
    if final_path is None:
        final_path = config_path / "final.json"

    if legacy:
        return AccessPoint(
            emodel_dir=config_path,
            final_path=final_path,
            legacy_dir_structure=True,
            with_seeds=True,
        )

    if not config_path.is_dir():
        return AccessPoint(
            nexus_config=config_path,
            emodel_dir=local_config,
            mech_path=Path(local_config) / "mechanisms",
        )

    return AccessPoint(
        emodel_dir=config_path.parent,
        final_path=final_path,
        recipes_path=config_path / "recipes.json",
        with_seeds=True,
    )


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("--no-progress", is_flag=True)
def cli(verbose, no_progress):
    """Cli."""
    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)]
    logformat = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=loglevel, format=logformat)

    if loglevel >= logging.INFO:  # pragma: no cover
        logging.getLogger("distributed").level = max(
            logging.getLogger("distributed").level, logging.WARNING
        )

    if no_progress:
        os.environ["NO_PROGRESS"] = "True"


@cli.command("compute_currents")
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--population_name", type=click.Path(exists=True), default=None)
@click.option("--output-path", default="circuit_currents.h5", type=str)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--hoc-path", type=str, required=True)
@click.option("--protocol-config-path", type=str, default=None)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--resume", is_flag=True)
@click.option("--sql-tmp-path", default=None, type=str)
@click.option("--debug-csv-path", default=None, type=str)
@click.option("--only-rin", is_flag=True)
def compute_currents(
    input_path,
    population_name,
    output_path,
    morphology_path,
    hoc_path,
    protocol_config_path,
    parallel_lib,
    resume,
    sql_tmp_path,
    debug_csv_path,
    only_rin,
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
    cells_df, input_cells = _load_circuit(input_path, morphology_path, population_name)

    if protocol_config_path is not None:
        with open(protocol_config_path, "r") as prot_file:
            protocol_config = yaml.safe_load(prot_file)
    else:
        protocol_config = {
            "holding_voltage": -85.0,
            "step_start": 500.0,
            "step_stop": 2000.0,
            "threshold_current_precision": 0.001,
            "min_threshold_current": 0.0,
            "max_threshold_current": 0.2,
            "spike_at_ais": False,  # does not work with placeholder
            "deterministic": True,
            "celsius": 34.0,
            "v_init": -80.0,
            "rin": {
                "with_ttx": False,  # if True, it requires TTXDynamicsSwitch mod file
                "step_start": 1000.0,
                "step_stop": 2000.0,
                "step_amp": -0.001,
            },
        }

    # we evaluate currents only for unique morph/emodel pairs
    unique_cells_df = evaluate_currents(
        cells_df.drop_duplicates(["morphology", "emodel"]),
        protocol_config,
        hoc_path,
        parallel_factory=parallel_factory,
        db_url=Path(sql_tmp_path) / "current_db.sql" if sql_tmp_path is not None else None,
        resume=resume,
        only_rin=only_rin,
    )

    cols = ["resting_potential", "input_resistance", "exception"]
    if not only_rin:
        cols += ["holding_current", "threshold_current"]

    # we populate the full circuit with duplicates if any
    if len(cells_df) == len(unique_cells_df):
        cells_df = unique_cells_df
    else:
        unique_cells_df = unique_cells_df.set_index(["morphology", "emodel"])
        for entry, data in tqdm(
            cells_df.groupby(["morphology", "emodel"]), disable=os.environ.get("NO_PROGRESS", False)
        ):
            for col in cols:
                cells_df.loc[data.index, col] = unique_cells_df.loc[entry, col]

    failed_cells = cells_df[
        cells_df["input_resistance"].isna() | (cells_df["input_resistance"] < 0)
    ].index
    if len(failed_cells) > 0:
        L.info("%s failed cells, we retry with fixed timesteps:", len(failed_cells))
        L.info(cells_df.loc[failed_cells])
        protocol_config["deterministic"] = False
        cells_df.loc[failed_cells] = evaluate_currents(
            cells_df.loc[failed_cells],
            protocol_config,
            hoc_path,
            parallel_factory=parallel_factory,
            db_url=Path(sql_tmp_path) / "current_db.sql" if sql_tmp_path is not None else None,
            resume=resume,
            only_rin=only_rin,
        )

        failed_cells = cells_df[
            cells_df["input_resistance"].isna() | (cells_df["input_resistance"] < 0)
        ].index
        if len(failed_cells) > 0:
            L.info("still %s failed cells (we drop):", len(failed_cells))
            L.info(cells_df.loc[failed_cells])
            cells_df.loc[failed_cells, "mtype"] = None

    cols_rename = {col: f"@dynamics:{col}" for col in cols if col != "exception"}

    if debug_csv_path is not None:
        cells_df.to_csv(debug_csv_path, index=False)

    # ensure we don't have any previous cols with different dtype that will result in duplicates
    for col in cols_rename.values():
        if col in cells_df.columns:
            cells_df = cells_df.drop(col, axis=1)

    cells_df = cells_df.rename(columns=cols_rename).drop(columns=["path", "exception", "emodel"])

    output_cells = CellCollection.from_dataframe(cells_df)
    output_cells.population_name = input_cells.population_name
    output_cells.remove_unassigned_cells()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_cells.save(output_path)

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
                cbar_kws={"label": "# worst feature", "shrink": 0.5},
            )
            plt.suptitle(f"emodel={emodel}")
            plt.tight_layout()
            _pdf.savefig()
            plt.close()

    L.info("Plotting feature distributions...")
    path = main_path / "feature_distributions"
    path.mkdir(exist_ok=True)
    for (emodel, mtype), df in tqdm(
        cells_df.groupby(["emodel", "mtype"]), disable=os.environ.get("NO_PROGRESS", False)
    ):
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
@click.option("--population_name", type=click.Path(exists=True), default=None)
@click.option("--output-path", default="evaluation_df.csv", type=str)
@click.option("--n-cells-per-emodel", default=None, type=int)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--config-path", type=str, required=True)
@click.option("--final-path", type=str, default=None)
@click.option("--local-config-path", type=str, default="config")
@click.option("--local-dir", type=str, default="local")
@click.option("--legacy", is_flag=True)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--resume", is_flag=True)
@click.option("--sql-tmp-path", default=None, type=str)
@click.option("--clip", default=5, type=float)
@click.option("--feature-filter", default="", type=str)
@click.option("--validation-path", default="analysis_plot", type=str)
@click.option("--with-model-management", is_flag=True)
@click.option("--evaluate-all", is_flag=True)
def evaluate(
    input_path,
    population_name,
    output_path,
    n_cells_per_emodel,
    morphology_path,
    config_path,
    final_path,
    local_config_path,
    local_dir,
    legacy,
    parallel_lib,
    resume,
    sql_tmp_path,
    clip,
    feature_filter,
    validation_path,
    with_model_management,
    evaluate_all,
):
    """Evaluate models from a circuit."""
    parallel_factory = init_parallel_factory(parallel_lib)
    access_point = _get_access_point(
        config_path, final_path, legacy, local_config=local_config_path
    )
    cells_df, _ = _load_circuit(input_path, morphology_path, population_name)
    # cells_df = cells_df[cells_df.emodel == "bAC_L6BTC"]
    # cells_df = cells_df[cells_df.mtype == "L23_LBC"].reset_index(drop=True)
    # cells_df["@dynamics:AIS_scaler"] = 4.0
    # cells_df["@dynamics:soma_scaler"] = 2.0

    if n_cells_per_emodel is not None:
        cells_df = (
            cells_df.groupby(["emodel", "mtype"])
            .sample(n_cells_per_emodel, random_state=42, replace=True)
            .reset_index(drop=True)
        )
    # remove category or it fails later sometimes in groupby
    for col in cells_df.columns:
        if cells_df[col].dtype == "category":
            cells_df[col] = cells_df[col].astype("object")

    # add data for adapted AIS/soma if available
    exemplar_data_path = Path(local_dir) / "exemplar_data.yaml"
    if Path(exemplar_data_path).exists():
        with open(exemplar_data_path) as exemplar_f:
            exemplar_data = yaml.safe_load(exemplar_f)

        for emodel, data in exemplar_data.items():
            if "ais" in data:
                cells_df.loc[cells_df.emodel == emodel, "ais_model"] = json.dumps(data["ais"])
                cells_df.loc[cells_df.emodel == emodel, "soma_model"] = json.dumps(data["soma"])
            cells_df = cells_df.rename(
                columns={
                    "@dynamics:AIS_scaler": "ais_scaler",
                    "@dynamics:soma_scaler": "soma_scaler",
                }
            )
    if not evaluate_all:
        if "ais_model" in cells_df.columns:
            cells_df = cells_df[~cells_df["ais_model"].isna()]
            L.info(
                "We found %s emodels with non-placeholders to evaluate.",
                len(cells_df.emodel.unique()),
            )
        else:
            L.info("Nothing to compute, only placeholder models found.")
            return

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

        exemplar_df = exemplar_df.set_index("emodel").loc[cells_df.emodel].reset_index()

        pass_dfs = []
        Path(validation_path).mkdir(parents=True, exist_ok=True)
        with PdfPages(Path(validation_path) / "mm_features.pdf") as pdf:
            for emodel in tqdm(cells_df.emodel.unique()):
                _cells_df = cells_df[cells_df.emodel == emodel]
                cells_score_df = get_score_df(_cells_df, filters=feature_filter)
                exemplar_score_df = get_score_df(
                    exemplar_df[exemplar_df.emodel == emodel], filters=feature_filter
                )

                _pass = cells_score_df.copy()
                for col in cells_score_df.columns:
                    _pass[col] = cells_score_df[col] <= np.maximum(
                        5.0, 5.0 * exemplar_score_df[col].to_list()[0]
                    )
                _pass = 1 - _pass.copy()  # copy to make it less fragmented
                _pass["mtype"] = cells_df.mtype

                data = _pass.groupby("mtype").mean()
                plt.figure(figsize=(10, 15))
                ax = plt.gca()
                sns.heatmap(
                    data.T,
                    xticklabels=True,
                    yticklabels=True,
                    vmin=0.0,
                    vmax=1.0,
                    ax=ax,
                    cbar_kws={"label": "Fraction of failed features", "shrink": 0.5},
                )

                ax.set_xlabel("mtype")
                ax.set_ylabel("feature")
                plt.suptitle(emodel)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                pass_df = pd.DataFrame()
                pass_df["pass"] = _pass.all(axis=1)
                pass_df["emodel"] = emodel
                pass_df["mtype"] = _cells_df.mtype
                pass_dfs.append(pass_df)
        pass_df = pd.concat(pass_dfs)
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

    parallel_factory.shutdown()


def compile_mechanisms(mech_path="mechanisms", compiled_mech_path=None):
    """Compile mechanisms in custom location."""
    if compiled_mech_path is None:
        compiled_mech_path = os.environ.get("TMPDIR", ".")
    cwd = os.getcwd()
    compiled_mech_path = Path(compiled_mech_path).resolve()
    compiled_mech_path.mkdir(exist_ok=True)
    mech_path = Path(mech_path).resolve()
    os.chdir(compiled_mech_path)
    subprocess.run(f"nrnivmodl {mech_path}", shell=True, check=True)
    os.chdir(cwd)


@cli.command("prepare")
@click.option("--config-path", type=str, required=True)
@click.option("--local-config-path", type=str, default="config")
@click.option("--mechanisms-path", type=str, required=True)
@click.option("--legacy", is_flag=True)
def prepare(config_path, local_config_path, mechanisms_path, legacy):
    """Prepare config folder and compile mechanisms."""
    access_point = _get_access_point(config_path, legacy=legacy, local_config=local_config_path)
    compile_mechanisms(access_point.mech_path, mechanisms_path)


@cli.command("assign")
@click.option("--input-node-path", type=click.Path(exists=True), required=True)
@click.option("--population_name", type=click.Path(exists=True), default=None)
@click.option("--output-node-path", default="node.h5", type=str)
@click.option("--config-path", type=str, required=True)
@click.option("--local-config-path", type=str, default="config")
def assign(input_node_path, population_name, output_node_path, config_path, local_config_path):
    """Assign emodels to cells in a circuit."""
    access_point = _get_access_point(config_path, local_config=local_config_path)
    cells_df, _ = _load_circuit(input_node_path, population_name=population_name)

    emodel_mappings = defaultdict(lambda: defaultdict(dict))
    L.info("Creating emodel mappings...")
    etype_emodel_map = None
    _config_path = config_path if Path(config_path).is_dir() else local_config_path
    if (Path(_config_path) / "etype_emodel_map.csv").exists():
        etype_emodel_map = pd.read_csv(Path(_config_path) / "etype_emodel_map.csv")

    if etype_emodel_map is not None:
        for (region, etype, mtype), d in etype_emodel_map.groupby(["region", "etype", "mtype"]):
            emodel_mappings[region][etype][mtype] = d["emodel"].values[0]  # assumes unique emodel
    else:
        for emodel in access_point.emodels:
            recipe = access_point.recipes[emodel]
            emodel_mappings[recipe["region"]][recipe["etype"]][recipe["mtype"]] = emodel

    def assign_emodel(row):
        """Get emodel name to use in pandas .apply."""
        try:
            return "hoc:" + emodel_mappings[row["region"]][row["etype"]][row["mtype"]]
        except KeyError:
            return "hoc:no_emodel"

    L.info("Assigning emodels...")
    cells_df["model_template"] = cells_df.apply(assign_emodel, axis=1)
    n_fails = len(cells_df[cells_df["model_template"] == "hoc:no_emodel"])
    if n_fails > 0:
        L.warning("%s cells could not be assigned emodels.", n_fails)

    L.info("Saving sonata file...")
    cells = CellCollection.load(input_node_path)
    cells.properties["model_template"] = cells_df["model_template"].to_list()
    cells.save(output_node_path)


@cli.command("adapt")
@click.option("--input-node-path", type=click.Path(exists=True), required=True)
@click.option("--population_name", type=click.Path(exists=True), default=None)
@click.option("--output-node-path", default="node.h5", type=str)
@click.option("--morphology-path", type=click.Path(exists=True), required=False)
@click.option("--config-path", type=str, required=True)
@click.option("--local-config-path", type=str, default="config")
@click.option("--final-path", type=str, default=None)
@click.option("--output-hoc-path", type=str, default="hoc")
@click.option("--local-dir", type=str, default="local")
@click.option("--legacy", is_flag=True)
@click.option("--parallel-lib", default="multiprocessing", type=str)
@click.option("--resume", is_flag=True)
@click.option("--sql-tmp-path", default=None, type=str)
@click.option("--min-scale", default=0.8, type=float)
@click.option("--max-scale", default=1.2, type=float)
def adapt(
    input_node_path,
    population_name,
    output_node_path,
    morphology_path,
    config_path,
    local_config_path,
    final_path,
    output_hoc_path,
    local_dir,
    legacy,
    parallel_lib,
    resume,
    sql_tmp_path,
    min_scale,
    max_scale,
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
    access_point = _get_access_point(
        config_path, final_path, legacy, local_config=local_config_path
    )

    local_dir = Path(local_dir)
    local_dir.mkdir(exist_ok=True, parents=True)

    cells_df, _ = _load_circuit(input_node_path, morphology_path, population_name)

    L.info("Extracting exemplar data...")

    exemplar_df = pd.DataFrame()
    for gid, emodel in enumerate(cells_df.emodel.unique()):
        if emodel != "no_emodel":
            morph = access_point.get_morphologies(emodel)
            exemplar_df.loc[gid, "emodel"] = emodel
            exemplar_df.loc[gid, "path"] = morph["path"]
            exemplar_df.loc[gid, "name"] = morph["name"]

    L.info("We found %s emodels.", len(exemplar_df))

    def _get_ais_profile(path):
        """Create ais profile from exemplar, instead of constant mean diameters as by default.

        This is important to remain closer to the exemplar morphology used for optimisation.
        """
        morphology = Morphology(path)
        orig_lens = next(extract_ais_path_distances([morphology]))
        orig_diams = next(extract_ais_diameters([morphology]))
        diams = np.interp(np.linspace(0, 60, 10), orig_lens, orig_diams)
        return diams.tolist()

    def _get_exemplar_data(exemplar_df):
        """Create exemplar data for all emodels."""
        exemplar_data = defaultdict(dict)
        _cached_data = {}  # used to avoid recomputing same exemplar data
        for emodel in tqdm(exemplar_df.emodel, disable=os.environ.get("NO_PROGRESS", False)):
            _df = exemplar_df[exemplar_df.emodel == emodel].copy()
            exemplar_path = _df["path"].tolist()[0]

            if len(Morphology(exemplar_path).root_sections) == 1:
                exemplar_data[emodel]["placeholder"] = True
            else:
                if exemplar_path not in _cached_data:
                    _df["mtype"] = "all"
                    _data = generate_exemplars(_df, with_plots=False, surface_percentile=50)
                    _data["ais"] = {"popt": _get_ais_profile(exemplar_path)}
                    _cached_data[exemplar_path] = _data

                exemplar_data[emodel] = _cached_data[exemplar_path]
                exemplar_data[emodel]["placeholder"] = False

        return dict(exemplar_data)

    with Reuse(local_dir / "exemplar_data.yaml") as reuse:
        exemplar_data = reuse(_get_exemplar_data, exemplar_df)

    L.info("Compute exemplar rho factors...")
    placeholder_mask = []
    for gid, emodel in enumerate(cells_df.emodel.unique()):
        if emodel != "no_emodel":
            if not exemplar_data[emodel]["placeholder"]:
                placeholder_mask.append(gid)
                exemplar_df.loc[gid, "ais_model"] = json.dumps(exemplar_data[emodel]["ais"])
                exemplar_df.loc[gid, "soma_model"] = json.dumps(exemplar_data[emodel]["soma"])
                exemplar_df.loc[gid, "soma_scaler"] = 1.0
                exemplar_df.loc[gid, "ais_scaler"] = 1.0

    n_placeholders = len(cells_df.emodel.unique()) - len(placeholder_mask)
    n_emodels = len(exemplar_df)
    L.info("We found %s placeholders models out of %s models.", n_placeholders, n_emodels)
    with Reuse(local_dir / "exemplar_rho.csv", index=False) as reuse:
        data = reuse(
            evaluate_rho,
            exemplar_df.loc[placeholder_mask],
            access_point,
            parallel_factory=parallel_factory,
            resume=resume,
            db_url=sql_tmp_path,
        )
        for col in data.columns:
            if col not in exemplar_df:
                exemplar_df[col] = None
            exemplar_df.loc[placeholder_mask, col] = data[col].to_list()

    with Reuse(local_dir / "exemplar_rho_axon.csv", index=False) as reuse:
        data = reuse(
            evaluate_rho_axon,
            exemplar_df.loc[placeholder_mask],
            access_point,
            parallel_factory=parallel_factory,
            resume=resume,
            db_url=sql_tmp_path,
        )
        for col in data.columns:
            if col not in exemplar_df:
                exemplar_df[col] = None
            exemplar_df.loc[placeholder_mask, col] = data[col].to_list()

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

    with Reuse(local_dir / "resistance_models.yaml") as reuse:
        resistance_models = reuse(
            _get_resistance_models, exemplar_df.loc[placeholder_mask], exemplar_data, scales_params
        )

    L.info("Adapting AIS and soma of all cells..")
    cells_df["ais_scaler"] = 0.0
    cells_df["soma_scaler"] = 0.0
    cells_df["ais_model"] = ""
    cells_df["soma_model"] = ""

    def _adapt():
        """Adapt AIS/soma scales to match the rho factors."""
        for emodel in tqdm(exemplar_df.emodel, disable=os.environ.get("NO_PROGRESS", False)):
            mask = cells_df["emodel"] == emodel

            if emodel in exemplar_data and not exemplar_data[emodel]["placeholder"]:
                L.info("Adapting a non placeholder model...")

                if len(Morphology(cells_df[mask].head(1)["path"].tolist()[0]).root_sections) == 1:
                    raise ValueError(
                        f"We cannot adapt the full emodel {emodel} to a placeholder morphology."
                    )

                cells_df.loc[mask, "ais_model"] = json.dumps(exemplar_data[emodel]["ais"])
                cells_df.loc[mask, "soma_model"] = json.dumps(exemplar_data[emodel]["soma"])

                rhos = (
                    exemplar_df[exemplar_df.emodel == emodel][["rho", "rho_axon", "emodel"]]
                    .set_index("emodel")
                    .to_dict()
                )
                cells_df.loc[mask] = adapt_soma_ais(
                    cells_df.loc[mask],
                    access_point,
                    resistance_models[emodel],
                    rhos,
                    parallel_factory=parallel_factory,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    n_steps=2,
                )

            else:
                if len(Morphology(cells_df[mask].head(1)["path"].tolist()[0]).root_sections) > 1:
                    raise ValueError(
                        f"We cannot adapt a placeholder emodel {emodel} to a full morphology."
                    )

        return cells_df

    with Reuse(local_dir / "adapt_df.csv") as reuse:
        cells_df = reuse(_adapt)

    # finally save a node.h5 file
    L.info("Saving sonata file...")
    cells = CellCollection.load(input_node_path)
    cells.properties["@dynamics:AIS_scaler"] = cells_df["ais_scaler"].to_list()
    cells.properties["@dynamics:soma_scaler"] = cells_df["soma_scaler"].to_list()
    cells.save(output_node_path)

    L.info("Create hoc files...")
    Path(output_hoc_path).mkdir(exist_ok=True)

    for emodel in exemplar_data:
        if not exemplar_data[emodel]["placeholder"]:
            hoc_params = [
                exemplar_data[emodel]["soma"]["soma_radius"],
                exemplar_data[emodel]["soma"]["soma_surface"],
            ] + exemplar_data[emodel]["ais"]["popt"]
            morph_modifier_hoc = [get_replace_axon_hoc(hoc_params)]
            template_path = _BASE_PATH / "templates" / "cell_template_neurodamus.jinja2"

        else:
            template_path = _BASE_PATH / "templates" / "cell_template_neurodamus_placeholder.jinja2"
            morph_modifier_hoc = []

        model_configuration = access_point.get_configuration(emodel)
        cell_model = create_cell_model(
            emodel,
            model_configuration=model_configuration,
            morph_modifiers=None,
            morph_modifiers_hoc=morph_modifier_hoc,
        )

        hoc = cell_model.create_hoc(
            access_point.final[emodel]["params"],
            template=template_path.name,
            template_dir=template_path.parent,
        )

        with open(Path(output_hoc_path) / f"{emodel}.hoc", "w") as hoc_file:
            hoc_file.write(hoc)

    parallel_factory.shutdown()
