from emodel_generalisation.access_point import AccessPoint
from bluepyopt.ephys.responses import TimeVoltageResponse

from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import numpy as np

from emodel_generalisation.utils import plot_traces
from hashlib import sha256
from itertools import cycle
import json
from functools import partial
import yaml
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from emodel_generalisation.mcmc import save_selected_emodels
from emodel_generalisation.modifiers import synth_axon
from emodel_generalisation.modifiers import synth_soma
from emodel_generalisation.evaluation import feature_evaluation
from emodel_generalisation.utils import get_score_df, get_combo_hash
from emodel_generalisation.mcmc import load_chains
from utils.reuse import Reuse


if __name__ == "__main__":
    mcmc_df = load_chains("run_df.csv")
    mask = mcmc_df.cost < 10
    # mask = mask & (mcmc_df['scores']['Step_ReboundBurstProtocol_burst_-300.soma.v.time_to_interburst_min']<1)
    # mcmc_df = mcmc_df[mcmc_df["normalized_parameters"]["gamma_TC_cad.alldend"] > 0.8]
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
