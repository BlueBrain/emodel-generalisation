from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from emodel_generalisation import ALL_LABELS
from emodel_generalisation.mcmc import load_chains

if __name__ == "__main__":
    path = Path("../cadpyr_l5")
    mcmc_df = load_chains(pd.read_csv(path / "out/cADpyr_L5/mcmc_df.csv"), base_path=path)
    mcmc_df = mcmc_df[mcmc_df.cost < 5].reset_index(drop=True)
    max_feat = mcmc_df["scores"].idxmax(axis=1).value_counts(ascending=False)
    # max_feat = max_feat.tail(20)
    print(max_feat)
    max_feat.index = [ALL_LABELS.get(p, p) for p in max_feat.index]

    plt.figure(figsize=(17, 7))
    max_feat.plot.bar(ax=plt.gca(), color="k")
    plt.yscale("log")
    plt.ylabel("# models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("worst_features.pdf")
