import json

import matplotlib.pyplot as plt
import pandas as pd
from datareuse import Reuse

from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import save_selected_emodels
from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.model.evaluation import feature_evaluation
from emodel_generalisation.utils import plot_traces

if __name__ == "__main__":
    mcmc_df = load_chains("run_df.csv")
    mask = mcmc_df.cost < 10
    mcmc_df = mcmc_df[mask].sort_values(by="cost")
    print(mcmc_df.cost)
    mcmc_df = mcmc_df.head(1).reset_index()
    emodel = "generic_model"
    save_selected_emodels(mcmc_df, mcmc_df.index, emodel=emodel, final_path="final.json")

    emodel_db = AccessPoint(
        emodel_dir=".",
        recipes_path="config/recipes.json",
        final_path="final.json",
        with_seeds=True,
    )
    emodel_db.morph_path = "simple.asc"
    df = pd.DataFrame()
    frozen_params = {}
    final = json.load(open("final.json"))
    for i, emodel in enumerate(final):
        df.loc[i, "name"] = f"{emodel}"
        df.loc[i, "emodel"] = emodel

    with Reuse("eval_df.csv") as reuse:
        df = reuse(
            feature_evaluation,
            df,
            emodel_db,
            parallel_factory="multiprocessing",
            trace_data_path="traces",
        )
    plot_traces(df)
    plt.close("all")
