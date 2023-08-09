"""Morphology related utils functions."""

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

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import neurom as nm
import numpy as np
import pandas as pd
import yaml
from diameter_synthesis import build_diameters
from diameter_synthesis import build_models
from diameter_synthesis.main import plot_models
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.morphdb import MorphDB
from morph_tool.resampling import resample_linear_density
from morphio.mut import Morphology
from neurom import NeuriteType
from neurom import view
from tqdm import tqdm

from emodel_generalisation.parallel.evaluator import evaluate


def create_combos_df(
    morphology_dataset_path, generalisation_rule_path, emodel, n_min_per_mtype, n_morphs
):
    """Create combo dataframe."""
    if Path(morphology_dataset_path).suffix == ".xml":
        combos_df = (
            MorphDB.from_neurondb(
                morphology_dataset_path, morphology_folder=Path(morphology_dataset_path).parent
            )
            .df[["name", "mtype", "layer", "path"]]
            .drop_duplicates("name")
        )
    if Path(morphology_dataset_path).suffix == ".csv":
        combos_df = pd.read_csv(morphology_dataset_path)

    def _class(mtype):
        """Eventually get this info from outside."""
        if "PC" in mtype:
            return "PC"
        else:
            return "IN"

    combos_df["morph_class"] = [_class(mtype) for mtype in combos_df.mtype]

    with open(generalisation_rule_path, "r") as f:
        generalisation_rule = yaml.safe_load(f)

    if "layer" in generalisation_rule:
        combos_df = combos_df[combos_df.layer == str(generalisation_rule["layer"])]
    if "morph_class" in generalisation_rule:
        combos_df = combos_df[combos_df.morph_class == generalisation_rule["morph_class"]]
    if "mtypes" in generalisation_rule:
        combos_df = combos_df[combos_df.mtype.isin(generalisation_rule["mtypes"])]

    combos_df["emodel"] = emodel
    combos_df["etype"] = generalisation_rule["etype"]
    for mtype in combos_df.mtype.unique():
        _df = combos_df[combos_df.mtype == mtype]
        if len(_df.index) < n_min_per_mtype:
            combos_df = combos_df.drop(index=_df.index).reset_index(drop=True)
    if n_morphs is not None:
        combos_df = combos_df.head(n_morphs)
    return combos_df


def _rediametrize(row, models, morphology_folder):
    """Rediametrizer and resampling to run in parallel."""
    neurite_types = ["basal_dendrite"]
    if row["morph_class"] == "PC":
        neurite_types.append("apical_dendrite")
    morph = Morphology(row["path"])
    morph = resample_linear_density(morph, 1.0)
    build_diameters.build(
        morph,
        neurite_types,
        models["simpler"][row["mtype"]],
        diam_params={"seed": 42, "models": ["simpler"]},
    )
    diametrized_path = morphology_folder / Path(row["path"]).name
    morph.write(diametrized_path)
    return {"diametrized_path": diametrized_path}


def plot_rediametrized(df, filename):
    """Plot original and rediametrized mmorphologies."""
    with PdfPages(filename) as pdf:
        for gid in tqdm(df.index):
            m_orig = nm.load_morphology(df.loc[gid, "orig_path"])
            m = nm.load_morphology(df.loc[gid, "path"])
            plt.figure()
            ax = plt.gca()

            m_orig = m_orig.transform(lambda x: x - np.array([200, 0, 0]))
            view.plot_morph(m_orig, ax, neurite_type=NeuriteType.basal_dendrite)
            view.plot_morph(m_orig, ax, neurite_type=NeuriteType.apical_dendrite)

            m = m.transform(lambda x: x + np.array([200, 0, 0]))
            view.plot_morph(m, ax, neurite_type=NeuriteType.basal_dendrite)
            view.plot_morph(m, ax, neurite_type=NeuriteType.apical_dendrite)

            plt.axis([-500, 500, -500, 1000])
            plt.axis("equal")
            plt.suptitle(df.loc[gid, "name"])
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def rediametrize(combo_df, out_folder, diameter_model_path, morphology_folder):
    """Rediametrize morphologies."""
    fig_folder = out_folder / "rediametrized_plot"
    fig_folder.mkdir(exist_ok=True)
    config_model = {"models": ["simpler"], "fig_folder": fig_folder}
    models = {"simpler": {}}
    data = {"simpler": {}}
    morphs = {}
    for mtype in tqdm(combo_df.mtype.unique()):
        neurite_types = ["basal_dendrite"]
        if np.unique(combo_df.loc[combo_df.mtype == mtype, "morph_class"])[0] == "PC":
            neurite_types.append("apical_dendrite")
        config_model["neurite_types"] = neurite_types
        morphs[mtype] = [
            nm.load_morphology(combo_df.loc[gid, "path"])
            for gid in combo_df[combo_df.mtype == mtype].index
        ]
        _model, _data = build_models.build(morphs[mtype], config_model, with_data=True)
        models["simpler"][mtype] = _model
        data["simpler"][mtype] = _data

    plot_models(morphs, config_model, models, data, ext=".pdf")
    with open(out_folder / diameter_model_path, "w") as f:
        yaml.dump(models, f)

    morphology_folder = out_folder / morphology_folder
    morphology_folder.mkdir(exist_ok=True, parents=True)

    combo_df = evaluate(
        combo_df,
        partial(_rediametrize, models=models, morphology_folder=morphology_folder),
        new_columns=[["diametrized_path", ""]],
        parallel_factory="multiprocessing",
    )
    combo_df["orig_path"] = combo_df["path"]
    combo_df["path"] = combo_df["diametrized_path"]
    plot_rediametrized(combo_df, filename=fig_folder / "rediametrized_morphs.pdf")
    return combo_df
