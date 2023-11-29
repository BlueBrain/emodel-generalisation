"""Convert nexus generate models to local AccessPoint config folder."""
import filecmp
import json
import logging
import os
import shutil
import subprocess
from copy import copy
from pathlib import Path

from tqdm import tqdm

L = logging.getLogger(__name__)


def _get_emodel_name(region, mtype, etype, i=None):
    """Create emodel name for internal use."""
    _region = copy(region)
    _mtype = copy(mtype)
    _etype = copy(etype)
    for char in [",", " ", "-", ":"]:
        _region = _region.replace(char, "_")
        _mtype = _mtype.replace(char, "_")
        _etype = _etype.replace(char, "_")
    base_name = f"{_region}__{_mtype}__{_etype}"
    if i is not None:
        return base_name + f"_{i}"
    return base_name


def _make_recipe_entry(config, emodel_name):
    with open(config["EModelPipelineSettings"]) as emodelsettings_file:
        emodelsettings = json.load(emodelsettings_file)

    with open(config["EModelConfiguration"]) as emodelconfig_file:
        emodelconfig = json.load(emodelconfig_file)

    path = Path(emodelconfig["morphology"].get("path"))
    return {
        "morph_path": str(path.parent),
        "morphology": str(path.name),
        "params": f"parameters/{emodel_name}.json",
        "features": f"features/{emodel_name}.json",
        "morph_modifiers": [],  # to update
        "pipeline_settings": {
            "efel_settings": emodelsettings["efel_settings"],
            "name_rmp_protocol": emodelsettings["name_rmp_protocol"],
            "name_Rin_protocol": emodelsettings["name_Rin_protocol"],
        },
    }


def _prepare(out_folder):
    """Prepare folder structure."""
    out_folder.mkdir(exist_ok=True)
    (out_folder / "parameters").mkdir(exist_ok=True)
    (out_folder / "features").mkdir(exist_ok=True)


def _make_mechanism(config, mech_path="mechanisms", base_path="."):
    """Copy mechanisms locally in a mechanisms folder."""
    mech_path = Path(mech_path)
    mech_path.mkdir(exist_ok=True, parents=True)
    with open(config["EModelConfiguration"]) as emodelconfig_file:
        emodelconfig = json.load(emodelconfig_file)

    for mech in emodelconfig["mechanisms"]:
        if mech["path"] is not None:
            local_mech_path = mech_path / Path(mech["path"]).name

            path = mech["path"]
            if not Path(mech["path"]).is_absolute():
                path = Path(base_path) / mech["path"]

            if local_mech_path.exists():
                if not filecmp.cmp(path, local_mech_path):
                    L.warning(
                        "Mechanism file %s  and %s are not the same,"
                        "but have the same name, we do not overwrite.",
                        mech["path"],
                        local_mech_path,
                    )
            else:
                shutil.copy(path, mech_path / Path(mech["path"]).name)


def _make_parameters(config):
    """Convert parameter entry."""
    entry = {"mechanisms": {}, "distributions": {}, "parameters": {}}
    with open(config["EModelConfiguration"]) as emodelconfig_file:
        emodelconfig = json.load(emodelconfig_file)

    for mech in emodelconfig["mechanisms"]:
        if mech["location"] not in entry["mechanisms"]:
            entry["mechanisms"][mech["location"]] = {"mech": []}
        entry["mechanisms"][mech["location"]]["mech"].append(mech["name"])

    for distr in emodelconfig["distributions"]:
        entry["distributions"][distr["name"]] = {"fun": distr["function"]}
        if "parameters" in distr:
            entry["distributions"][distr["name"]]["parameters"] = distr["parameters"]

    for param in emodelconfig["parameters"]:
        if param["location"] not in entry["parameters"]:
            entry["parameters"][param["location"]] = []
        entry["parameters"][param["location"]].append(
            {"name": param["name"], "val": param["value"]}
        )
    return entry


def _make_features(config):
    """Convert features entry."""
    with open(config["FitnessCalculatorConfiguration"]) as fitnessconf_file:
        fitnessconf = json.load(fitnessconf_file)
    return fitnessconf


def _make_parameter_entry(config):
    """Convert model parameters data.

    Only the parameters are considered, the others are not needed.
    """
    with open(config["EModel"]) as emodel_file:
        emodel = json.load(emodel_file)

    entry = {"params": {}}
    for param in emodel["parameter"]:
        entry["params"][param["name"]] = param["value"]
    return entry


def _add_emodel(
    recipes,
    final,
    region,
    mtype,
    etype,
    config,
    i,
    out_config_folder,
    mech_path="mechanisms",
    base_path=".",
):
    """Add a single emodel."""
    emodel_name = _get_emodel_name(region, mtype, etype, i)
    recipes[emodel_name] = _make_recipe_entry(config, emodel_name)
    recipes[emodel_name]["etype"] = etype
    recipes[emodel_name]["mtype"] = mtype
    recipes[emodel_name]["region"] = region
    final[emodel_name] = _make_parameter_entry(config)

    params = _make_parameters(config)
    _make_mechanism(config, mech_path, base_path)

    with open(out_config_folder / recipes[emodel_name]["params"], "w") as param_file:
        json.dump(params, param_file, indent=4)

    features = _make_features(config)
    with open(out_config_folder / recipes[emodel_name]["features"], "w") as feat_file:
        json.dump(features, feat_file, indent=4)


def convert_all_config(config_path, out_config_folder="config", mech_path="mechanisms"):
    """Convert a nexus config_json file into a local config folder loadable via AccessPoint."""
    with open(config_path) as config_file:
        config = json.load(config_file)

    out_config_folder = Path(out_config_folder)
    _prepare(out_config_folder)
    recipes = {}
    final = {}
    for region, region_config in tqdm(config.items()):
        for mtype, mtype_config in region_config.items():
            for etype, etype_config in mtype_config.items():
                for entry, data in etype_config["eModel"].items():
                    if not Path(data).is_absolute():
                        etype_config["eModel"][entry] = Path(config_path).parent / data
                if etype_config["assignmentAlgorithm"] == "assignOne":
                    _add_emodel(
                        recipes,
                        final,
                        region,
                        mtype,
                        etype,
                        etype_config["eModel"],
                        None,
                        out_config_folder,
                        mech_path,
                        Path(config_path).parent,
                    )

    with open(out_config_folder / "recipes.json", "w") as recipes_file:
        json.dump(recipes, recipes_file, indent=4)

    with open(out_config_folder / "final.json", "w") as final_file:
        json.dump(final, final_file, indent=4)


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
