from pathlib import Path

import matplotlib.pyplot as plt

from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import plot_corner
from emodel_generalisation.mcmc import plot_cost
from emodel_generalisation.mcmc import plot_cost_convergence

if __name__ == "__main__":
    df = load_chains("run_df.csv")
    df.loc[df.cost > 5, "cost"] = 5
    plot_cost_convergence(df)

    split = 2.0

    max_feat = df[df.cost < split]["scores"].idxmax(axis=1).value_counts(ascending=True)
    plt.figure(figsize=(7, 9))
    max_feat.plot.barh(ax=plt.gca())
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("worst_features.pdf")

    plot_cost(df, split, filename="costs.pdf")
    df = df[df.cost < split].reset_index(drop=True)

    _df = df[df.cost < split].reset_index(drop=True)

    plot_corner(
        _df,
        feature=None,
        filename="corner.pdf",
    )

    plot_corner(
        _df,
        feature="cost",
        filename="corner_cost.pdf",
    )
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
