from pathlib import Path
import pytest
import pandas as pd

from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.parallel import init_parallel_factory

DATA = Path(__file__).parent / "data"


@pytest.fixture
def simple_morph():
    return str(DATA / "simple.asc")


@pytest.fixture
def morphs_combos_df(simple_morph):
    df = pd.DataFrame()
    df.loc[0, "name"] = "simple"
    df.loc[0, "path"] = simple_morph
    df.loc[0, "emodel"] = "generic_model"
    return df


@pytest.fixture
def access_point():
    return AccessPoint(
        emodel_dir=DATA,
        recipes_path=DATA / "config" / "recipes.json",
        final_path=DATA / "final.json",
    )
