"""Convert nexus generate models to local AccessPoint config folder."""
import json
from pathlib import Path


def _get_emodel_name(region, mtype, etype, i):
    """Create emodel name for internal use."""
    return f"{region}_{mtype}_{etype}_{i}"


def _make_recipe_entry(config, emodel_name):
    with open(config["EModelPipelineSettings"]) as emodelsettings_file:
        emodelsettings = json.load(emodelsettings_file)

    with open(config["EModelConfiguration"]) as emodelconfig_file:
        emodelconfig = json.load(emodelconfig_file)

    return {
        "morph_path": emodelconfig["morphology"].get("path", "."),  # to update
        "morphology": emodelconfig["morphology"]["name"],
        "params": f"config/parameters/{emodel_name}.json",
        "features": f"config/features/{emodel_name}.json",
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


def convert_all_config(config_path, out_config_folder="config"):
    """Convert a nexus config_json file into a local config folder loadable via AccessPoint."""
    with open(config_path) as config_file:
        config = json.load(config_file)

    out_config_folder = Path(out_config_folder)
    _prepare(out_config_folder)
    recipes = {}
    emodel_parameters = {}
    for region, region_config in config.items():
        for mtype, mtype_config in region_config.items():
            for etype, etype_config in mtype_config.items():
                for i, config in enumerate(etype_config):
                    emodel_name = _get_emodel_name(region, mtype, etype, i)
                    recipes[emodel_name] = _make_recipe_entry(config, emodel_name)
                    emodel_parameters[emodel_name] = _make_parameter_entry(config)

                    params = _make_parameters(config)
                    with open(recipes[emodel_name]["params"], "w") as param_file:
                        json.dump(params, param_file, indent=4)

                    features = _make_features(config)
                    with open(recipes[emodel_name]["features"], "w") as feat_file:
                        json.dump(features, feat_file, indent=4)

    with open(out_config_folder / "recipes.json", "w") as recipes_file:
        json.dump(recipes, recipes_file, indent=4)

    with open(out_config_folder / "emodel_parameters.json", "w") as emodel_params_file:
        json.dump(emodel_parameters, emodel_params_file, indent=4)
