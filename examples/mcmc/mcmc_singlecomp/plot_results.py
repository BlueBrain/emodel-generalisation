from pathlib import Path

import matplotlib.pyplot as plt

from emodel_generalisation.information import plot_corner
from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import plot_cost
from emodel_generalisation.mcmc import plot_cost_convergence

if __name__ == "__main__":
    df = load_chains("run_df.csv")
    df.loc[df.cost > 10, "cost"] = 10
    print(df)
    plot_cost_convergence(df)

    split = 5.0

    max_feat = df[df.cost < split]["scores"].idxmax(axis=1).value_counts(ascending=True)
    plt.figure(figsize=(7, 9))
    max_feat.plot.barh(ax=plt.gca())
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("worst_features.pdf")

    plot_cost(df, split, filename="costs.pdf")
    df = df[df.cost < split].reset_index(drop=True)

    _df = df[df.cost < split].reset_index(drop=True)
    best_id = df.cost.idxmin()
    print(df.loc[best_id, "cost"])

    plt.figure(figsize=(5, 3))
    plt.axvline(30, c="k", label="best")
    plt.axvline(31, c="k", ls="--", label="1sd")
    plt.axvline(29, c="k", ls="--")
    plt.axvline(35, c="k", ls="-.", label="5sd")
    plt.axvline(25, c="k", ls="-.")
    plt.axvline(
        df.loc[best_id, "features"]["IDRest_200.soma.v.mean_frequency"], c="r", label="best model"
    )
    plt.hist(df["features"]["IDRest_200.soma.v.mean_frequency"], bins=100, label="MCMC")
    plt.legend(loc="best")
    plt.xlabel("mean frequency IDRest 200")
    plt.ylabel("number of models")
    plt.tight_layout()
    plt.savefig("mean_freq_200.png")

    plt.figure(figsize=(5, 3))
    plt.axvline(45, c="k", label="best")
    plt.axvline(46, c="k", ls="--", label="1sd")
    plt.axvline(44, c="k", ls="--")
    plt.axvline(50, c="k", ls="-.", label="5sd")
    plt.axvline(40, c="k", ls="-.")
    plt.axvline(
        df.loc[best_id, "features"]["IDRest_300.soma.v.mean_frequency"], c="r", label="best model"
    )
    plt.hist(df["features"]["IDRest_300.soma.v.mean_frequency"], bins=100, label="MCMC")
    # plt.legend(loc="best")
    plt.xlabel("mean frequency IDRest 300")
    plt.ylabel("number of models")
    plt.tight_layout()
    plt.savefig("mean_freq_300.png")

    plot_corner(_df, feature=None, filename="corner.pdf", highlights=[[best_id], ["r"]])

    plot_corner(_df, feature="cost", filename="corner_cost.pdf", highlights=[[best_id], ["r"]])

    Path("corners").mkdir(exist_ok=True)
    for feature in _df["scores"].mean(axis=0).sort_values(ascending=False).index:
        print("corner plot of ", feature)
        plot_corner(
            _df.reset_index(drop=True),
            feature=("scores", feature),
            # feature=("features", feature),
            filename=f"corners/corner_{feature}.pdf",
        )
        plt.close()
