import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from currentscape.currentscape import plot_currentscape
from datareuse import Reuse

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.mcmc import bin_data
from emodel_generalisation.mcmc import get_2d_correlations
from emodel_generalisation.mcmc import get_mean_sd
from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import plot_corner
from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.model.evaluation import feature_evaluation

currents = [
    "i_pas",
    "ihcn_Ih",
    "ica_Ca_HVA2",
    "ica_Ca_LVAst",
    "ik_SK_E2",
    "ik_SKv3_1",
    "ik_K_Pst",
    "ik_K_Tst",
    "ina_NaTg",
]


if __name__ == "__main__":
    path = Path("../cadpyr_l5")
    mcmc_df = load_chains(pd.read_csv(path / "out/cADpyr_L5/mcmc_df.csv"), base_path=path)
    mcmc_df = mcmc_df[mcmc_df.cost < 4].reset_index(drop=True)
    protocols = json.load(
        open(path / "out/cADpyr_L5/configs/cADpyr_L5/config/protocols/cADpyr_L5PC.json")
    )
    protocols["Step_200"]["extra_recordings"] = []
    for var in currents:
        protocols["Step_200"]["extra_recordings"].append(
            {
                "var": var,
                "comp_x": 0.5,
                "type": "nrnseclistcomp",
                "name": "soma",
                "seclist_name": "soma",
                "sec_index": 0,
            }
        )
    json.dump(
        protocols,
        open(path / "out/cADpyr_L5/configs/cADpyr_L5/config/protocols/cADpyr_L5PC.json", "w"),
        indent=4,
    )
    emodel = "cADpyr_L5"

    access_point = AccessPoint(
        emodel_dir=path / "out/cADpyr_L5/configs",
        final_path=path / "out/cADpyr_L5/final.json",
        with_seeds=True,
        legacy_dir_structure=True,
    )

    efeatures = access_point.get_json(emodel, "features")

    features = [
        "SearchThresholdCurrent.soma.v.bpo_threshold_current",
        "SearchHoldingCurrent.soma.v.bpo_holding_current",
        "Step_200.soma.v.AP_amplitude",
        "Step_200.soma.v.mean_frequency",
        "Step_200.soma.v.AHP_depth",
        "Step_200.soma.v.inv_time_to_first_spike",
        "Step_200.soma.v.inv_first_ISI",
    ]

    # select some models
    mcmc_df = mcmc_df.sort_values(by=("features", features[2])).reset_index(drop=True)
    model_ids = [
        int(0.105 * len(mcmc_df)),
        int(len(mcmc_df) * 0.5),
        int(len(mcmc_df) * 0.5) + 900,
        int(len(mcmc_df) * 0.905),
    ]
    print(model_ids)
    model_colors = ["g", "r", "m", "b"]

    plt.figure()
    fig, axs = plt.subplots(1, len(features), figsize=(1.5 * len(features), 3))
    for ax, feat in zip(axs, features):
        f_mean, f_std = get_mean_sd(efeatures, feat)
        d = mcmc_df["features"][feat]
        ax.hist(d, bins=30, orientation="horizontal", color="k", histtype="step")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_ylabel(ALL_LABELS.get(feat, feat))
        ax.set_xlabel("# models")
        for i, c in zip(model_ids, model_colors):
            ax.axhline(mcmc_df.loc[i]["features"][feat], c=c)

        ax.axhline(f_mean, c="k", label="exp. mean")
        ax.axhline(f_mean - f_std, c="k", ls="--")
        ax.axhline(f_mean + f_std, c="k", ls="--", label="exp. 1sd")
        ax.axhline(f_mean - 2 * f_std, c="k", ls="-.")
        ax.axhline(f_mean + 2 * f_std, c="k", ls="-.", label="exp. 2sd")
    plt.tight_layout()
    plt.savefig("feature_hist.pdf")

    _df = mcmc_df.copy()
    _df = mcmc_df.drop(
        columns=[c for c in mcmc_df.columns if (c[0] == "features") and (c[1] not in features)]
    )

    pf_cor = get_2d_correlations(_df, y_col="features", tpe="pearson")
    best_corr_threshold = 0.2
    pf_cor = (
        pf_cor[abs(pf_cor) > best_corr_threshold]
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="all")
    )
    # pf_cor[pf_cor > 0.5] = 0.5

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    pf_cor.index = [ALL_LABELS.get(p, p) for p in pf_cor.index]
    pf_cor.columns = [ALL_LABELS.get(p, p) for p in pf_cor.columns]
    pf_cor.to_csv("feature_param_cor.csv")
    sns.heatmap(
        data=pf_cor.T,
        ax=ax,
        vmin=-1,  # best_corr_threshold,
        vmax=1,
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="k",
        cbar_kws={"label": "MI", "shrink": 0.3},
        xticklabels=True,
        yticklabels=True,
        square=True,
    )
    x = pf_cor.to_numpy()
    ids = np.array(np.argwhere(abs(x) > best_corr_threshold), dtype=float)
    ids += 0.5
    ax.scatter(*ids.T, marker="o", c="k", s=5.0)
    plt.tight_layout()

    plt.savefig("param_feature_MI.pdf")

    plt.figure(figsize=(4, 2))
    m = bin_data(
        mcmc_df["parameters"]["gIhbar_Ih.somadend"],
        mcmc_df["features"]["SearchHoldingCurrent.soma.v.bpo_holding_current"],
        np.ones(len(mcmc_df)),
        n=30,
        mode="sum",
        _min1=mcmc_df["parameters"]["gIhbar_Ih.somadend"].min(),
        _max1=mcmc_df["parameters"]["gIhbar_Ih.somadend"].max(),
        _min2=mcmc_df["features"]["SearchHoldingCurrent.soma.v.bpo_holding_current"].min(),
        _max2=mcmc_df["features"]["SearchHoldingCurrent.soma.v.bpo_holding_current"].max(),
    )
    plt.imshow(
        m.T,
        aspect="auto",
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        vmin=0,
        extent=[
            mcmc_df["parameters"]["gIhbar_Ih.somadend"].min(),
            mcmc_df["parameters"]["gIhbar_Ih.somadend"].max(),
            mcmc_df["features"]["SearchHoldingCurrent.soma.v.bpo_holding_current"].min(),
            mcmc_df["features"]["SearchHoldingCurrent.soma.v.bpo_holding_current"].max(),
        ],
    )
    plt.xlabel(ALL_LABELS.get("gIhbar_Ih.somadend"))
    plt.ylabel(ALL_LABELS.get("SearchHoldingCurrent.soma.v.bpo_holding_current"))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.colorbar(label="# models", shrink=0.3)
    for i, c in zip(model_ids, model_colors):
        plt.scatter(
            mcmc_df.loc[i]["parameters"]["gIhbar_Ih.somadend"],
            mcmc_df.loc[i]["features"]["SearchHoldingCurrent.soma.v.bpo_holding_current"],
            c=c,
            s=20,
        )

    plt.tight_layout()
    plt.savefig("holding.pdf")

    plt.figure(figsize=(4, 2))
    m = bin_data(
        mcmc_df["parameters"]["gNaTgbar_NaTg.somatic"],
        mcmc_df["features"]["Step_200.soma.v.AP_amplitude"],
        np.ones(len(mcmc_df)),
        n=30,
        mode="sum",
        _min1=mcmc_df["parameters"]["gNaTgbar_NaTg.somatic"].min(),
        _max1=mcmc_df["parameters"]["gNaTgbar_NaTg.somatic"].max(),
        _min2=mcmc_df["features"]["Step_200.soma.v.AP_amplitude"].min(),
        _max2=mcmc_df["features"]["Step_200.soma.v.AP_amplitude"].max(),
    )
    plt.imshow(
        m.T,
        aspect="auto",
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        vmin=0,
        extent=[
            mcmc_df["parameters"]["gNaTgbar_NaTg.somatic"].min(),
            mcmc_df["parameters"]["gNaTgbar_NaTg.somatic"].max(),
            mcmc_df["features"]["Step_200.soma.v.AP_amplitude"].min(),
            mcmc_df["features"]["Step_200.soma.v.AP_amplitude"].max(),
        ],
    )

    plt.xlabel(ALL_LABELS.get("gNaTgbar_NaTg.somatic"))
    plt.ylabel(ALL_LABELS.get("Step_200.soma.v.AP_amplitude"))
    plt.colorbar(label="# models", shrink=0.3)
    for i, c in zip(model_ids, model_colors):
        plt.scatter(
            mcmc_df.loc[i]["parameters"]["gNaTgbar_NaTg.somatic"],
            mcmc_df.loc[i]["features"]["Step_200.soma.v.AP_amplitude"],
            c=c,
            s=20,
        )
    plt.tight_layout()
    plt.savefig("ap_amp.pdf")

    pp_cor = get_2d_correlations(mcmc_df)
    best_corr_threshold = 0.03

    _df = pp_cor[pp_cor > best_corr_threshold].dropna(axis=1, how="all").dropna(axis=0, how="all")
    params = list(set(_df.columns.to_list() + _df.index.to_list()))
    _df = mcmc_df.drop(
        columns=[
            c for c in mcmc_df.columns if (c[0] == "normalized_parameters") and (c[1] not in params)
        ]
    )

    plot_corner(
        _df,
        filename="corner.pdf",
        n_bins=12,
        cmap="Greys",
        normalize=True,
        highlights=[model_ids, model_colors],
    )

    exemplar_data = yaml.safe_load(open(path / "out/cADpyr_L5/exemplar_models.yaml"))
    combos_df = pd.DataFrame()
    for i, c in zip(model_ids, model_colors):
        combos_df.loc[i, "new_parameters"] = json.dumps(mcmc_df.loc[i, "parameters"].to_dict())
        combos_df.loc[i, "path"] = str(path / exemplar_data["paths"]["all"])
        combos_df.loc[i, "name"] = c
    combos_df["ais_model"] = json.dumps(exemplar_data["ais"])
    combos_df["ais_scaler"] = 1.0
    combos_df["soma_model"] = json.dumps(exemplar_data["soma"])
    combos_df["soma_scaler"] = 1.0
    combos_df["emodel"] = "cADpyr_L5_9084"

    with Reuse("features.csv") as reuse:
        combos_df = reuse(
            feature_evaluation,
            combos_df,
            access_point,
            parallel_factory="multiprocessing",
            trace_data_path="traces",
        )
    for index in combos_df.index:
        trace_path = combos_df.loc[index, "trace_data"]
        plt.figure(figsize=(5, 2))
        with open(f"../cadpyr_l5/mcmc_data/{trace_path}", "rb") as f:
            trace = pickle.load(f)[1]
            response = trace["Step_200.soma.v"]
            plt.plot(response["time"], response["voltage"], c=combos_df.loc[index, "name"], lw=1)
            plt.axis([500, 3000, -90, 40])

        plt.savefig(f"trace_{index}.pdf")

    config = {"current": {"names": currents}, "output": {"savefig": True, "extension": "pdf"}}
    for index in combos_df.index:
        trace_path = combos_df.loc[index, "trace_data"]
        plt.figure(figsize=(5, 2))
        with open(f"../cadpyr_l5/mcmc_data/{trace_path}", "rb") as f:
            trace = pickle.load(f)[1]
            response = trace["Step_200.soma.v"]
            time = response["time"]
            voltage = response["voltage"]
            data_currents = []
            ions_data = []
            for var in currents:
                data_currents.append(trace[f"Step_200.soma.{var}"]["voltage"])
        config["output"]["fname"] = f"current_scape_{index}"
        config["output"]["extension"] = "png"  # pdf are large, to save space
        plot_currentscape(voltage, data_currents, config, time=time)
