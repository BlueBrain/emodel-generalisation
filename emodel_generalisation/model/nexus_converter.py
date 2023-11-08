"""Convert nexus generate models to local AccessPoint config folder."""
import json
from pathlib import Path
from tqdm import tqdm


def _get_emodel_name(region, mtype, etype, i=None):
    """Create emodel name for internal use."""
    base_name = f"{region.replace(',', '').replace(' ', '_')}__{mtype}__{etype}"
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
    with open(
        config.get(
            "FitnessCalculatorConfiguration",
            "/gpfs/bbp.cscs.ch/data/project/proj136/nexus/bbp/mmb-emodels-for-synthesized-neurons/e/d/a/8/d/7/f/7/FCC__emodel=cNAC__etype=cNAC__iteration=fa285b7.json",
        )
    ) as fitnessconf_file:
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


def _add_emodel(recipes, final, region, mtype, etype, config, i, out_config_folder):
    """Add a single emodel."""
    emodel_name = _get_emodel_name(region, mtype, etype, i)
    recipes[emodel_name] = _make_recipe_entry(config, emodel_name)
    recipes[emodel_name]["etype"] = etype
    recipes[emodel_name]["mtype"] = mtype
    recipes[emodel_name]["region"] = region
    final[emodel_name] = _make_parameter_entry(config)

    params = _make_parameters(config)
    with open(out_config_folder / recipes[emodel_name]["params"], "w") as param_file:
        json.dump(params, param_file, indent=4)

    features = _make_features(config)
    with open(out_config_folder / recipes[emodel_name]["features"], "w") as feat_file:
        json.dump(features, feat_file, indent=4)


def convert_all_config(config_path, out_config_folder="config"):
    """Convert a nexus config_json file into a local config folder loadable via AccessPoint."""
    with open(config_path) as config_file:
        config = json.load(config_file)

    out_config_folder = Path(out_config_folder)
    _prepare(out_config_folder)
    recipes = {}
    final = {}
    for region, region_config in tqdm(list(config.items())):
        for mtype, mtype_config in region_config.items():
            for etype, etype_config in mtype_config.items():
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
                    )

    with open(out_config_folder / "recipes.json", "w") as recipes_file:
        json.dump(recipes, recipes_file, indent=4)

    with open(out_config_folder / "final.json", "w") as final_file:
        json.dump(final, final_file, indent=4)
