from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.mcmc import get_mean_sd
from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.utils import get_feature_df

if __name__ == "__main__":
    path = Path("../cadpyr_l5")
    df = pd.read_csv(path / "out/cADpyr_L5/evaluations.csv").set_index(["name", "emodel"])
    df_no = pd.read_csv(path / "out/cADpyr_L5/evaluations_no_adaptation.csv").set_index(
        ["name", "emodel"]
    )
    df_no = df_no.loc[df.index]
    emodel = "cADpyr_L5_9084"
    access_point = AccessPoint(
        emodel_dir=path / "out/cADpyr_L5/configs",
        final_path=path / "out/cADpyr_L5/final.json",
        with_seeds=True,
        legacy_dir_structure=True,
    )

    efeatures = access_point.get_json(emodel, "features")
    d = get_feature_df(df)
    d_no = get_feature_df(df_no)

    print(d)
    print(d_no)

    features = [
        "SearchThresholdCurrent.soma.v.bpo_threshold_current",
        "SearchHoldingCurrent.soma.v.bpo_holding_current",
        "Step_200.soma.v.AP_amplitude",
        "Step_200.soma.v.mean_frequency",
        "Step_200.soma.v.AHP_depth",
        "Step_200.soma.v.inv_time_to_first_spike",
        "Step_200.soma.v.inv_first_ISI",
    ]

    bs = [
        (0, 2),
        (-0.5, 0.2),
        (20, 120),
        (0, 50),
        (0, 50),
        (0, 150),
        (0, 130),
    ]
    f_mean, f_std = get_mean_sd(efeatures, features[0])
    print("threshold reduction:")
    print(len(d[d[features[0]] > f_mean + 5 * f_std]), len(d))
    print(len(d_no[d_no[features[0]] > f_mean + 5 * f_std]), len(d_no))
    fig, axs = plt.subplots(len(features), 1, figsize=(4, 1.3 * len(features)))
    for ax, feat, b in zip(axs, features, bs):
        ax.hist(
            np.clip(d[feat], b[0], b[1]),
            histtype="step",
            bins=50,
            label="adapted",
            density=True,
            # orientation="horizontal",
            log=True,
            color="k",
        )
        ax.hist(
            np.clip(d_no[feat], b[0], b[1]),
            histtype="step",
            bins=50,
            label="not adapted",
            density=True,
            # orientation="horizontal",
            log=True,
            color="xkcd:azure",
        )
        f_mean, f_std = get_mean_sd(efeatures, feat)

        ax.axvline(f_mean, c="k")
        ax.axvline(f_mean - f_std, c="k", ls="--")
        ax.axvline(f_mean + f_std, c="k", ls="--")
        ax.axvline(f_mean - 2 * f_std, c="k", ls="-.")
        ax.axvline(f_mean + 2 * f_std, c="k", ls="-.")
        ax.axvline(f_mean - 5 * f_std, c="k", ls="dotted")
        ax.axvline(f_mean + 5 * f_std, c="k", ls="dotted")

        ax.set_xlim([0.95 * b[0], b[1] * 1.05])
        ax.set_xlabel(ALL_LABELS.get(feat))
        ax.set_ylabel("density of models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("feature_comparison.pdf")
