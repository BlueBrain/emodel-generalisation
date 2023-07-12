"""
Copyright (c) 2022 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode
or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

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
