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

"""Pytest conftest."""
from pathlib import Path

import pandas as pd
import pytest

from emodel_generalisation.model.access_point import AccessPoint

DATA = Path(__file__).parent / "data"
# pylint: disable=redefined-outer-name


@pytest.fixture
def simple_morph():
    """ """
    return str(DATA / "simple.asc")


@pytest.fixture
def morphs_combos_df(simple_morph):
    """ """
    df = pd.DataFrame()
    df.loc[0, "name"] = "simple"
    df.loc[0, "path"] = simple_morph
    df.loc[0, "emodel"] = "generic_model"
    return df


@pytest.fixture
def access_point():
    """ """
    return AccessPoint(
        emodel_dir=DATA,
        recipes_path=DATA / "config" / "recipes.json",
        final_path=DATA / "final.json",
    )
