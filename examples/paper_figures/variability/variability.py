from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.mcmc import get_mean_sd
from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_feature_df
from emodel_generalisation.utils import get_scores

if __name__ == "__main__":
    path = Path("../cadpyr_l5")
    df = pd.read_csv(path / "out/cADpyr_L5/evaluations.csv")
    mtype = "L5_TPC:A"
    emodel = "cADpyr_L5"
    selected = yaml.safe_load(open(path / f"out/cADpyr_L5/selected/selected_{mtype}.yaml"))
    exemplar_data = yaml.safe_load(open(path / "out/cADpyr_L5/exemplar_models.yaml"))
    exp_df = pd.read_csv("../ephys_plot/exp_features.csv", index_col=0).T
    exp_df = exp_df.rename(
        columns={
            "Threshold current": "SearchThresholdCurrent.soma.v.bpo_threshold_current",
            "Holding current": "SearchHoldingCurrent.soma.v.bpo_holding_current",
        }
    )
    features = [
        "SearchThresholdCurrent.soma.v.bpo_threshold_current",
        "SearchHoldingCurrent.soma.v.bpo_holding_current",
        "Step_200.soma.v.AP_amplitude",
        "Step_200.soma.v.mean_frequency",
        "Step_200.soma.v.AHP_depth",
        "Step_200.soma.v.inv_time_to_first_spike",
        "Step_200.soma.v.inv_first_ISI",
    ]
    access_point = AccessPoint(
        legacy_dir_structure=True,
        with_seeds=True,
        final_path=path / "out/cADpyr_L5/final.json",
        emodel_dir=path / "out/cADpyr_L5/configs/",
    )

    efeatures = access_point.get_json(emodel, "features")
    df = df[df.mtype == mtype]
    df = get_scores(df, features_to_ignore=FEATURE_FILTER)
    emodels = selected["emodels"]
    morphs = selected["morphos"]
    _clip = 1e-9
    test = partial(stats.levene, center="mean")
    d = pd.DataFrame()
    gid = 0
    fig, axs = plt.subplots(1, len(features), figsize=(1.5 * len(features), 3))
    all_p = pd.DataFrame()
    for f, ax in zip(features, axs):
        f_mean, f_std = get_mean_sd(efeatures, f)
        _exp = exp_df[f].to_numpy()
        _exp = _exp[~np.isnan(_exp)]

        _df = df[df.cost < 5]
        all_feat_df = get_feature_df(_df[_df.emodel.isin(emodels) & _df.name.isin(morphs)]).sample(
            500
        )
        all_p.loc[f, "pvalue"] = np.clip(test(all_feat_df[f].to_numpy(), _exp).pvalue, _clip, 1)
        all_p.loc[f, "feature"] = ALL_LABELS.get(f, f)
        for emodel in emodels:
            e_df = df[df.emodel == emodel]
            e_df = e_df.set_index("name")
            e_df = e_df.loc[morphs]
            e_feat_df = get_feature_df(e_df.reset_index())

            if emodel == "cADpyr_L5_8825":
                _e_f = e_feat_df[f].to_list()

            e_pvalue = np.clip(test(e_feat_df[f].to_numpy(), _exp).pvalue, _clip, 1)
            d.loc[gid, "pvalue"] = e_pvalue
            d.loc[gid, "type"] = "fix model"
            d.loc[gid, "feature"] = ALL_LABELS.get(f, f)
            gid += 1

        for morph in morphs:
            m_df = df[df.name == morph]
            m_df = m_df.set_index("emodel")
            m_df = m_df.loc[emodels]
            m_feat_df = get_feature_df(m_df.reset_index())
            m_f = m_feat_df[f].to_list()
            if morph == "vd110530_idC":
                _m_f = m_feat_df[f].to_list()

            m_pvalue = np.clip(test(m_feat_df[f].to_numpy(), _exp).pvalue, _clip, 1)
            d.loc[gid, "pvalue"] = m_pvalue
            d.loc[gid, "type"] = "fix morphology"
            d.loc[gid, "feature"] = ALL_LABELS.get(f, f)
            gid += 1

        _d = pd.DataFrame()
        i = 0
        for _e in _e_f:
            _d.loc[i, "value"] = _e
            _d.loc[i, "type"] = "fix model"
            _d.loc[i, "feature"] = f
            i += 1
        for _m in _m_f:
            _d.loc[i, "value"] = _m
            _d.loc[i, "type"] = "fix morphology"
            _d.loc[i, "feature"] = f
            i += 1
        for _f in all_feat_df[f]:
            _d.loc[i, "value"] = _f
            _d.loc[i, "type"] = "all"
            _d.loc[i, "feature"] = f
            i += 1

        sns.stripplot(
            data=_d,
            x="feature",
            y="value",
            hue="type",
            dodge=True,
            ax=ax,
            palette={"fix model": "maroon", "fix morphology": "olivedrab", "all": "k"},
            size=3,
        )

        ax.axhline(f_mean, c="k")
        ax.axhline(f_mean + f_std, c="k", ls="--")
        ax.axhline(f_mean - f_std, c="k", ls="--")
        ax.axhline(f_mean + 2 * f_std, c="k", ls="-.")
        ax.axhline(f_mean - 2 * f_std, c="k", ls="-.")
        ax.axhline(_d.loc[np.argwhere(np.array(morphs) == "vd110530_idC")[0][0], "value"], c="r")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_ylabel(ALL_LABELS.get(f))
        ax.set_xlabel("")
        ax.xaxis.set_ticklabels([])
        ax.get_legend().remove()

    plt.legend()
    plt.tight_layout()
    plt.savefig("comp_distributions.pdf")

    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    sns.stripplot(
        data=d,
        y="feature",
        x="pvalue",
        hue="type",
        orient="h",
        dodge=True,
        size=4,
        ax=ax,
        linewidth=0,
        # color="k",
        palette={"fix model": "maroon", "fix morphology": "olivedrab", "all": "k"},
    )
    sns.stripplot(
        data=all_p,
        y="feature",
        x="pvalue",
        orient="h",
        size=8,
        ax=ax,
        linewidth=0,
        color="k",
    )
    plt.axvline(0.05, c="k", ls="--")
    ax.set_xlim(_clip, 1.05)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("variability.pdf")
