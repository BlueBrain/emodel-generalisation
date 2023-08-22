"""Test cli module."""
import numpy.testing as npt
from pathlib import Path
import pandas as pd

from voxcell import CellCollection

import emodel_generalisation.cli as tested

DATA = Path(__file__).parent / "data"


def test_compute_currents(cli_runner, tmpdir):
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "compute_currents",
            "--input-path", DATA / "sonata_v6.h5",
            "--output-path", tmpdir / "sonata_currents.h5",
            "--morphology-path", str(DATA / "morphologies"),
            "--protocol-config-path", str(DATA / "protocol_config.yaml"),
            "--hoc-path", str(DATA / "hoc"),
            "--parallel-lib", None,
            "--debug-csv-path", "debug.csv"  # use this to debug
        ],
    )
    # fmt: on
    assert response.exit_code == 0

    df = CellCollection().load_sonata(tmpdir / "sonata_currents.h5").as_dataframe()
    npt.assert_allclose(
        df["@dynamics:resting_potential"].to_list(),
        [-78.24665843577513, -78.04757822491321],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:input_resistance"].to_list(),
        [239.4410588137958, 200.66982375732323],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:holding_current"].to_list(),
        [-0.028562604715887, -0.035378993149493],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:threshold_current"].to_list(), [0.08203125, 0.1109375], rtol=1e-5
    )


def test_evaluate(cli_runner, tmpdir):
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "evaluate",
            "--input-path", DATA / "sonata_v6.h5",
            "--output-path", tmpdir / "evaluation_df.csv",
            "--morphology-path", str(DATA / "morphologies"),
            "--config-path", str(DATA / "config"),
            "--final-path", str(DATA / "final.json"),
            "--parallel-lib", None,
        ],
    )
    # fmt: on

    assert response.exit_code == 0

    df = pd.read_csv(tmpdir / "evaluation_df.csv")
    df.to_csv(DATA / "evaluation_df.csv", index=None)
    expected_df = pd.read_csv(DATA / "evaluation_df.csv")

    assert df.loc[0, "features"] == expected_df.loc[0, "features"]
    assert df.loc[0, "scores"] == expected_df.loc[0, "scores"]
    assert df.loc[1, "features"] == expected_df.loc[1, "features"]
    assert df.loc[1, "scores"] == expected_df.loc[1, "scores"]




def test_adapt(cli_runner, tmpdir):
    tmpdir = Path('tmp')
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "adapt",
            "--input-node-path", DATA / "sonata_v6.h5",
            "--output-csv-path", tmpdir / "adapt_df.csv",
            "--output-node-path", tmpdir / "sonata_v6_adapted.h5",
            "--morphology-path", str(DATA / "morphologies"),
            "--config-path", str(DATA / "config"),
            "--final-path", str(DATA / "final.json"),
            "--hoc-path", str(tmpdir / "hoc"),
            "--parallel-lib", None,
        ],
    )
    # fmt: on

    assert response.exit_code == 0

    df = pd.read_csv(tmpdir / "evaluation_df.csv")
    #df.to_csv(DATA / "evaluation_df.csv", index=None)
    #expected_df = pd.read_csv(DATA / "evaluation_df.csv")
