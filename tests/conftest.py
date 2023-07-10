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
