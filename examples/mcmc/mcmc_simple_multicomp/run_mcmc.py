"""Test MCM module."""
from pathlib import Path
import pandas as pd
from emodel_generalisation import mcmc

from emodel_generalisation.model.access_point import AccessPoint


if __name__ == "__main__":
    access_point = AccessPoint(
        emodel_dir=".", recipes_path="config/recipes.json", final_path="final.json"
    )
    access_point.morph_path = "simple.asc"
    mcmc.run_several_chains(
        proposal_params={"std": 0.1},
        temperature=0.5,
        n_chains=80,
        n_steps=100,
        results_df_path="chains",
        run_df_path="run_df.csv",
        access_point=access_point,
        emodel="generic_model",
        random_initial_parameters=True,
    )
