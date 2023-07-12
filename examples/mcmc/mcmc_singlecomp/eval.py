import matplotlib.pyplot as plt
import pandas as pd
from utils.reuse import Reuse

from emodel_generalisation.access_point import AccessPoint
from emodel_generalisation.evaluation import feature_evaluation
from emodel_generalisation.mcmc import load_chains
from emodel_generalisation.mcmc import save_selected_emodels
from emodel_generalisation.utils import plot_traces

if __name__ == "__main__":
    mcmc_df = load_chains("run_df.csv")
    mask = mcmc_df.cost < 10
    mcmc_df = mcmc_df[mask].sort_values(by="cost")
    print(mcmc_df.cost)
    mcmc_df = mcmc_df.head(1).reset_index()
    emodel = "all_generic"
    save_selected_emodels(mcmc_df, mcmc_df.index, emodel=emodel, final_path="final.json")
    emodel = "all_generic_0"

    emodel_db = AccessPoint(
        emodel_dir=".",
        recipes_path="config/recipes.json",
        final_path="final.json",
        with_seeds=True,
        legacy_dir_structure=False,
    )
    df = pd.DataFrame()
    df.loc[0, "name"] = f"{emodel}"
    df.loc[0, "emodel"] = emodel
    print(df)
    with Reuse("test.csv") as reuse:
        df = reuse(
            feature_evaluation,
            df,
            emodel_db,
            parallel_factory="multiprocessing",
            trace_data_path="traces",
            # record_ions_and_currents=True,
        )
    plot_traces(df)
    plt.close("all")
