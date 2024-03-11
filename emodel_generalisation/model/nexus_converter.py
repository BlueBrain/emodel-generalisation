"""Convert nexus generate models to local AccessPoint config folder."""
import filecmp
import logging
import shutil
from copy import copy
from pathlib import Path

import pandas as pd

from emodel_generalisation.utils import load_json
from emodel_generalisation.utils import write_json

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


def _make_recipe_entry(config):
    morph_file = Path(config["morphology"])

    recipe = {
        "morph_path": str(morph_file.parent),
        "morphology": morph_file.name,
        "params": config["params"]["bounds"],
        "features": config.get("features", None),
        "morph_modifiers": [],  # to update
    }

    if config["pipeline_settings"] is not None:
        emodelsettings = load_json(config["pipeline_settings"])
        recipe["pipeline_settings"] = {
            "efel_settings": emodelsettings["efel_settings"],
            "name_rmp_protocol": emodelsettings["name_rmp_protocol"],
            "name_Rin_protocol": emodelsettings["name_Rin_protocol"],
        }
    else:
        recipe["pipeline_settings"] = None

    return recipe


def _make_mechanisms(mechanisms: list, mech_path="mechanisms", base_path="."):
    """Copy mechanisms locally in a mechanisms folder."""
    mech_path = Path(mech_path)
    mech_path.mkdir(exist_ok=True, parents=True)

    for mech in mechanisms:
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


def _make_parameter_entry(emodel_values):
    """Convert model parameters data.

    Only the parameters are considered, the others are not needed.
    """
    return {"params": {param["name"]: param["value"] for param in emodel_values["parameter"]}}


def _add_emodel(
    config,
    mech_path="mechanisms",
    base_path=".",
):
    param_values_path = config["params"]["values"]
    param_bounds_path = config["params"]["bounds"]

    final = _make_parameter_entry(load_json(param_values_path))
    recipe = _make_recipe_entry(config)

    emodel_configuration = load_json(param_bounds_path)
    _make_mechanisms(
        mechanisms=emodel_configuration["mechanisms"],
        mech_path=mech_path,
        base_path=base_path,
    )
    return final, recipe


def convert_all_config(config_path, out_config_folder="config", mech_path="mechanisms"):
    """Convert a nexus config_json file into a local config folder loadable via AccessPoint."""
    configuration = load_json(config_path)

    out_config_folder = Path(out_config_folder)
    out_config_folder.mkdir(exist_ok=True)

    base_config_dir = Path(config_path).parent

    final = {}
    recipes = {}
    for name, emodel_config in configuration["library"]["eModel"].items():
        emodel_config = _resolve_paths(emodel_config, base_config_dir)

        final[name], recipes[name] = _add_emodel(
            emodel_config,
            mech_path=mech_path,
            base_path=base_config_dir,
        )

    write_json(filepath=out_config_folder / "recipes.json", data=recipes, indent=4)
    write_json(filepath=out_config_folder / "final.json", data=final, indent=4)

    data = {"region": [], "etype": [], "mtype": [], "emodel": []}
    for region, data_region in configuration["configuration"].items():
        for mtype, data_mtype in data_region.items():
            for etype, data_etype in data_mtype.items():
                data["region"].append(region)
                data["etype"].append(etype)
                data["mtype"].append(mtype)
                data["emodel"].append(data_etype["eModel"])

    etype_emodel_map_df = pd.DataFrame.from_dict(data)
    etype_emodel_map_df.to_csv(out_config_folder / "etype_emodel_map.csv", index=False)


def _resolve_paths(emodel_config: dict, base_dir: Path) -> dict:
    """Resolve relative paths with respect to the base dir."""
    new_config = {}
    for k, v in emodel_config.items():
        if isinstance(v, dict):
            new_config[k] = _resolve_paths(v, base_dir)
        else:
            # if v is absolute, concatenation drops the prefix
            new_config[k] = str(base_dir / v)
    return new_config
