"""Module to adapt AIS/soma scale by matching rho factors."""

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
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from numpy.polynomial import Polynomial
from scipy.signal import cspline2d

from emodel_generalisation.model.evaluation import evaluate_ais_rin
from emodel_generalisation.model.evaluation import evaluate_rho
from emodel_generalisation.model.evaluation import evaluate_rho_axon
from emodel_generalisation.model.evaluation import evaluate_soma_rin
from emodel_generalisation.model.evaluation import feature_evaluation
from emodel_generalisation.model.evaluation import rin_evaluation
from emodel_generalisation.model.modifiers import remove_axon
from emodel_generalisation.model.modifiers import remove_soma
from emodel_generalisation.parallel import evaluate
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_scores

matplotlib.use("Agg")


def get_scales(scales_params, with_unity=False):
    """Create scale array from parameters."""
    if scales_params["lin"]:
        scales = np.linspace(scales_params["min"], scales_params["max"], scales_params["n"])
    else:
        scales = np.logspace(scales_params["min"], scales_params["max"], scales_params["n"])

    if with_unity:
        return np.insert(scales, 0, 1)
    return scales


def build_resistance_models(
    access_point,
    emodels,
    exemplar_data,
    scales_params,
    rcond_min=1e-2,
    key="ais",
):
    """Build resistance model of AIS/soma."""
    scales = get_scales(scales_params)

    df = pd.DataFrame()
    i = 0
    for emodel in emodels:
        for scale in scales:
            df.loc[i, "emodel"] = emodel
            df.loc[i, f"{key}_scaler"] = scale
            i += 1
    df["path"] = exemplar_data["paths"]["all"]
    df[f"{key}_model"] = json.dumps(exemplar_data[key])

    if key == "ais":
        func = evaluate_ais_rin
    if key == "soma":
        func = evaluate_soma_rin

    # it seems dask does not quite work on this (to investigate, but multiprocessing is fast enough)
    df = func(df, access_point, parallel_factory="multiprocessing")

    models = {}
    for emodel in emodels:
        _df = df[df.emodel == emodel]
        scaler = _df[f"{key}_scaler"].to_numpy()
        rin = _df[f"rin_{key}"].to_numpy()
        if len(rin[rin < 0]) == 0:
            try:
                coeffs, extra = Polynomial.fit(np.log10(scaler), np.log10(rin), 3, full=True)
                if extra[0] < rcond_min:
                    models[emodel] = {
                        "resistance": {"polyfit_params": coeffs.convert().coef.tolist()},
                        "shape": exemplar_data[key],
                    }
            except (np.linalg.LinAlgError, TypeError):
                print(f"fail to fit emodel {emodel}")
    return df[df.emodel.isin(models)], models


def plot_resistance_models(fit_df, models, pdf_filename="resistance_model.pdf", key="ais"):
    """Plot resistance models."""
    emodels = fit_df.emodel.unique()
    with PdfPages(pdf_filename) as pdf:
        for emodel in emodels:
            mask = fit_df.emodel == emodel
            plt.figure(figsize=(5, 3))
            fit_df[mask].plot(x=f"{key}_scaler", y=f"rin_{key}", marker="+", ax=plt.gca())
            plt.plot(
                fit_df[mask][f"{key}_scaler"],
                10
                ** Polynomial(models[emodel]["resistance"]["polyfit_params"])(
                    np.log10(fit_df[mask][f"{key}_scaler"])
                ),
                "-o",
                ms=0.5,
                label="fit",
            )
            plt.yscale("log")
            plt.xscale("log")
            plt.legend()
            plt.suptitle(emodel)
            plt.ylabel(f"{key} input resistance")

            pdf.savefig(bbox_inches="tight")
            plt.close()


def _adapt_combo(combo, models, rhos, key="soma", min_scale=0.01, max_scale=10):
    """Adapt soma/ais size."""
    emodel = combo["emodel"]
    rin = combo[f"rin_no_{key}"] * rhos[emodel]
    p = Polynomial(models[emodel]["resistance"]["polyfit_params"])

    roots_all = (p - np.log10(rin)).roots()
    roots = roots_all[np.imag(roots_all) == 0]
    scale = 10 ** np.real(roots[np.argmin(abs(roots - 1))])

    return {f"{key}_scaler": np.clip(scale, min_scale, max_scale)}


def _adapt_single_soma_ais(
    combo,
    access_point=None,
    models=None,
    rhos=None,
    n_steps=3,
    min_scale=0.01,
    max_scale=10,
):
    for _ in range(n_steps):
        combo.update(
            rin_evaluation(
                combo,
                access_point=access_point,
                morph_modifiers=[remove_axon],
                key="rin_no_ais",
                ais_recording=False,
            )
        )
        if combo["rin_no_ais"] is not None:
            combo.update(
                _adapt_combo(
                    combo,
                    models["ais"],
                    rhos["rho_axon"],
                    key="ais",
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
            )
        combo.update(
            rin_evaluation(
                combo,
                access_point=access_point,
                morph_modifiers=[remove_soma],
                key="rin_no_soma",
                ais_recording=True,
            )
        )
        if combo["rin_no_soma"] is not None:
            combo.update(
                _adapt_combo(
                    combo,
                    models["soma"],
                    rhos["rho"],
                    key="soma",
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
            )

    return {k: combo[k] for k in ["soma_scaler", "ais_scaler"]}


def make_evaluation_df(combos_df, emodels, exemplar_data, rhos=None):
    """Make a df to be evaluated."""
    df = pd.DataFrame()
    for emodel in emodels:
        if rhos is not None:
            if emodel in rhos["rho"]:
                _df = combos_df.copy()
                _df["emodel"] = emodel
                df = pd.concat([df, _df]).reset_index(drop=True)
        else:
            _df = combos_df.copy()
            _df["emodel"] = emodel
            df = pd.concat([df, _df]).reset_index(drop=True)

    df["ais_model"] = json.dumps(exemplar_data["ais"])
    df["ais_scaler"] = 1.0
    df["soma_model"] = json.dumps(exemplar_data["soma"])
    df["soma_scaler"] = 1.0

    return df


def adapt_soma_ais(
    df,
    access_point,
    models,
    rhos,
    parallel_factory,
    n_steps=2,
    min_scale=0.01,
    max_scale=10.0,
):
    """Adapt soma and ais size to match rho factors."""
    return evaluate(
        df,
        partial(
            _adapt_single_soma_ais,
            access_point=access_point,
            models=models,
            rhos=rhos,
            n_steps=n_steps,
            min_scale=min_scale,
            max_scale=max_scale,
        ),
        new_columns=[["soma_scaler", 1.0], ["ais_scaler", 1.0]],
        parallel_factory=parallel_factory,
        db_url=None,
    )


def build_all_resistance_models(access_point, emodels, exemplar_data, scales_params, fig_path=None):
    """Build resistance models of AIS and soma."""
    fit_df, ais_models = build_resistance_models(
        access_point, emodels, exemplar_data, scales_params, key="ais"
    )
    if fig_path is not None:
        plot_resistance_models(
            fit_df, ais_models, pdf_filename=fig_path / "ais_resistance_model.pdf", key="ais"
        )

    emodels = list(ais_models.keys())  # don't compute for failed emodels on ais
    fit_df, soma_models = build_resistance_models(
        access_point, emodels, exemplar_data, scales_params, key="soma"
    )

    if fig_path is not None:
        plot_resistance_models(
            fit_df,
            soma_models,
            pdf_filename=fig_path / "soma_resistance_model.pdf",
            key="soma",
        )
    ais_models = {k: ais_models[k] for k in soma_models}  # remove failed emodels on soma
    return {"soma": soma_models, "ais": ais_models}


RHO_FILTER = [
    "ohmic_input_resistance_vb_ssse",
    "bpo_holding_current",
    "bpo_threshold_current",
    "mean_frequency",
]


def create_rho_config(access_point, access_point_new):
    """Create bap valildation config."""
    protocols = access_point.get_json("protocol")
    new_prot_path = access_point_new.get_json_path("protocol")
    protocols["Main"]["other_protocols"] = [
        "Main",
        "RinHoldcurrent",
        "ThresholdDetection",
        "Step_200",
    ]

    with open(new_prot_path, "w") as f:
        json.dump(protocols, f, indent=4)

    features = access_point.get_json("features")
    new_feat_path = access_point_new.get_json_path("features")
    new_features = {}
    prots_to_keep = ["Rin", "RMP", "RinHoldCurrent", "Threshold", "Step_150"]
    for prot, feature in features.items():
        if prot in prots_to_keep:
            new_features[prot] = feature
    with open(new_feat_path, "w") as f:
        json.dump(new_features, f, indent=4)


def find_rho_factors(
    emodels,
    exemplar_data,
    mtype,
    access_point,
    parallel_factory,
    out_path,
    clip=5,
    ais_scales=None,
    soma_scales=None,
):
    """Find rho factors."""
    # hardcoded, but need to be passed from outside
    scan_folder = out_path / "rhos_scan"
    scan_folder.mkdir(parents=True, exist_ok=True)
    if ais_scales is None:
        ais_scales = np.linspace(0.5, 1.5, 10)
    if soma_scales is None:
        soma_scales = np.linspace(0.5, 1.5, 10)
    dfs = []
    for emodel in emodels:
        _df = pd.DataFrame()

        for i, (a, s) in enumerate(itertools.product(ais_scales, soma_scales)):
            _df.loc[i, "ais_scaler"] = a
            _df.loc[i, "soma_scaler"] = s

        _df["path"] = exemplar_data["paths"][mtype]
        _df["name"] = Path(exemplar_data["paths"][mtype]).stem
        _df["ais_model"] = json.dumps(exemplar_data["ais"])
        _df["soma_model"] = json.dumps(exemplar_data["soma"])
        _df["emodel"] = emodel
        dfs.append(_df)

    df = pd.concat(dfs).reset_index(drop=True)

    print("computing features")
    df = feature_evaluation(df, access_point, parallel_factory=parallel_factory)
    print("computing rho_axon")
    df = evaluate_rho_axon(df, access_point, parallel_factory=parallel_factory)

    print("computing rho")
    df = evaluate_rho(df, access_point, parallel_factory=parallel_factory)

    df.to_csv(scan_folder / f"rho_scan_{mtype}.csv")

    # df = pd.read_csv(scan_folder / f"rho_scan_{mtype}.csv")

    df = get_scores(df)
    for gid in df.index:
        _s = 0
        try:
            for f, s in df.loc[gid, "scores"].items():
                if not any(f.startswith(_f) for _f in FEATURE_FILTER):
                    _s = max(_s, np.clip(s, 0, clip))
        except AttributeError:
            _s = clip

        df.loc[gid, "score"] = _s
    scan_folder.mkdir(exist_ok=True)
    target_rhos = {"rho": {}, "rho_axon": {}}
    with PdfPages(scan_folder / f"rhos_{mtype}.pdf") as pdf:
        for emodel in df.emodel.unique():
            _df = df[df.emodel == emodel]
            pivot_df = _df.pivot(index="ais_scaler", columns="soma_scaler", values="score")
            pivot_df.loc[:, :] = cspline2d(pivot_df.to_numpy(), 1.0, 0.1)
            m = pivot_df.idxmin(axis=1)
            _id = np.argmin([pivot_df.loc[i, m[i]] for i in m.index])
            best_ais_scaler = m.index[_id]
            best_soma_scaler = m.iloc[_id]
            best = _df[(_df.ais_scaler == best_ais_scaler) & (_df.soma_scaler == best_soma_scaler)]
            rho = float(best["rho"].to_list()[0])
            rho_axon = float(best["rho_axon"].to_list()[0])
            if rho > 0 and rho_axon > 0:
                target_rhos["rho"][emodel] = rho
                target_rhos["rho_axon"][emodel] = rho_axon

                plt.figure()
                sns.heatmap(data=pivot_df, ax=plt.gca())
                plt.scatter(
                    np.argwhere(pivot_df.columns == best_soma_scaler)[0][0] + 0.5,
                    np.argwhere(pivot_df.index == best_ais_scaler)[0][0] + 0.5,
                    c="r",
                )
                plt.suptitle(f"emodel={emodel}, rho={rho}, rho_axon={rho_axon}")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
    return target_rhos
