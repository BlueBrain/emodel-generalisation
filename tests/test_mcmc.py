"""Test MCM module."""
from pathlib import Path

import pandas as pd
from numpy.testing import assert_almost_equal

from emodel_generalisation import mcmc

DATA = Path(__file__).parent / "data"


def test_mcmc(access_point, simple_morph, tmpdir):
    """ """
    access_point.morph_path = simple_morph
    mcmc.run_several_chains(
        proposal_params={"std": 0.1},
        temperature=0.5,
        n_chains=4,
        n_steps=3,
        results_df_path=tmpdir / "chains",
        run_df_path=tmpdir / "run_df.csv",
        access_point=access_point,
        random_initial_parameters=True,
    )
    df = mcmc.load_chains(str(tmpdir / "run_df.csv"))
    # df.to_csv(DATA / "mcmc_df.csv")
    expected_df = pd.read_csv(DATA / "mcmc_df.csv", header=[0, 1])
    for col in ["features", "parameters", "scores", "normalized_parameters"]:
        for feat in df[col].columns:
            assert_almost_equal(
                df[col][feat].to_numpy(), expected_df[col][feat].to_numpy(), decimal=0
            )
