import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bluepyefe import igorpy

from emodel_generalisation import ALL_LABELS

if __name__ == "__main__":
    # main_path = Path("/gpfs/bbp.cscs.ch/home/arnaudon/code/singlecell-features/cADpyr/cADpyr")
    # data_path = Path(
    #    "/gpfs/bbp.cscs.ch/project/proj38/singlecell/expdata/LNMC/_data/from_neobase_db1/"
    # )
    main_path = data_path = Path("../cadpyr_l5/ephys_data")
    cells = {
        "C061128B1-SR-C1": "high freq",
        "C060202A1-SR-C1": "low freq",
        "C061128A3-SR-C1": "mean freq",
    }

    def read(cell, _id):
        prot = json.load(open(main_path / cell / "protocols.json"))["Step_200"]["stimuli"]["step"]
        try:
            path = data_path / cell / f"exp_IDrest_ch0_{_id}.ibw"
            path1 = data_path / cell / f"exp_IDrest_ch1_{_id}.ibw"
            _, current = igorpy.read(path)
            data, voltage = igorpy.read(path1)
        except FileNotFoundError:
            path = data_path / cell / f"X_IDrest_ch0_{_id}.ibw"
            path1 = data_path / cell / f"X_IDrest_ch1_{_id}.ibw"
            _, current = igorpy.read(path)
            data, voltage = igorpy.read(path1)
        """
        import shutil

        Path(f"../cadpyr_l5/ephys_data/{cell}").mkdir(exist_ok=True)
        shutil.copy(
            main_path / cell / "protocols.json", f"../cadpyr_l5/ephys_data/{cell}/protocols.json"
        )
        shutil.copy(path, f"../cadpyr_l5/ephys_data/{cell}/{path.name}")
        shutil.copy(path1, f"../cadpyr_l5/ephys_data/{cell}/{path1.name}")
        """

        current *= 1e9
        voltage *= 1e3
        t = np.arange(0, data.dx * len(voltage), data.dx)
        t *= 1e3
        return t, voltage, current, prot

    for cell in cells:
        plt.figure(figsize=(5, 2))
        diff = []
        ids = []
        for _id in range(0, 1000):
            try:
                t, voltage, current, prot = read(cell, _id)
                curr = np.mean(
                    current[(t > prot["delay"]) & (t < prot["delay"] + prot["duration"])]
                )
                diff.append(abs(prot["amp"] - curr))
                ids.append(_id)
            except FileNotFoundError:
                pass
        t, voltage, current, prot = read(cell, ids[np.argmin(diff)])
        plt.plot(t, voltage, lw=0.8, c="k")
        plt.axis([0, 3000, -90, 35])
        plt.suptitle(cells[cell])
        plt.xlabel("ms")
        plt.ylabel("mV")
        plt.savefig(f"trace_{cells[cell]}.pdf")
    thresholds = json.load(open(main_path / "hypamp_threshold.json"))
    df = pd.DataFrame()
    for cell in main_path.iterdir():
        if cell.is_dir():
            feat = json.load(open(cell / "features.json"))
            """
            import shutil

            Path(f"../cadpyr_l5/ephys_data/{cell.stem}").mkdir(exist_ok=True)
            shutil.copy(
                cell / "features.json", f"../cadpyr_l5/ephys_data/{cell.stem}/features.json"
            )
            """

            df.loc["Threshold current", cell.stem] = thresholds[cell.stem]["threshold"]
            df.loc["Holding current", cell.stem] = thresholds[cell.stem]["hypamp"]
            for prot in feat:
                for loc in feat[prot]:
                    for f in feat[prot][loc]:
                        df.loc[".".join([prot, loc, f["feature"]]), cell.stem] = f["val"][0]
    print(df)
    df.to_csv("exp_features.csv")
    print(df.loc["Step_200.soma.v.mean_frequency"].sort_values())
    print(df.loc["Step_200.soma.v.mean_frequency"].sort_values().mean())
    features = [
        "Threshold current",
        "Holding current",
        "Step_200.soma.v.AP_amplitude",
        "Step_200.soma.v.mean_frequency",
        "Step_200.soma.v.AHP_depth",
        "Step_200.soma.v.inv_time_to_first_spike",
        "Step_200.soma.v.inv_first_ISI",
    ]
    df = df.loc[features]
    plt.figure()
    fig, axs = plt.subplots(1, len(features), figsize=(1.5 * len(features), 3))
    for ax, feat in zip(axs, features):
        d = df.loc[feat]
        sns.stripplot(data=d, ax=ax, color="k")
        sns.stripplot(data=d[[list(cells.keys())[0]]], ax=ax, color="r", size=8)
        sns.stripplot(data=d[[list(cells.keys())[1]]], ax=ax, color="g", size=8)
        sns.stripplot(data=d[[list(cells.keys())[2]]], ax=ax, color="b", size=8)
        ax.axhline(d.mean(), c="k")
        ax.axhline(d.mean() + d.std(), c="k", ls="--")
        ax.axhline(d.mean() - d.std(), c="k", ls="--")
        ax.axhline(d.mean() + 2 * d.std(), c="k", ls="-.")
        ax.axhline(d.mean() - 2 * d.std(), c="k", ls="-.")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_ylabel(ALL_LABELS.get(feat, feat))
        ax.xaxis.set_ticklabels([])
    plt.tight_layout()
    plt.savefig("ephys.pdf")
