"""Test cli module."""
import json
from pathlib import Path

import numpy.testing as npt
import pandas as pd
from voxcell import CellCollection

import emodel_generalisation.cli as tested

DATA = Path(__file__).parent / "data"


def test_compute_currents(cli_runner, tmpdir):
    """Test cli compute_currents."""
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "-v", "compute_currents",
            "--input-path", str(DATA / "sonata_v6.h5"),
            "--output-path", str(tmpdir / "sonata_currents.h5"),
            "--morphology-path", str(DATA / "morphologies"),
            "--protocol-config-path", str(DATA / "protocol_config.yaml"),
            "--hoc-path", str(DATA / "hoc"),
            "--parallel-lib", None,
        ],
    )
    # fmt: on
    assert response.exit_code == 0

    df = CellCollection().load_sonata(tmpdir / "sonata_currents.h5").as_dataframe()
    expected_rmp = [-78.18732146256286, -78.01157423898272]
    expected_rin = [244.72829316246703, 202.20371382555413]
    npt.assert_allclose(df["@dynamics:resting_potential"].to_list(), expected_rmp, rtol=1e-5)
    npt.assert_allclose(df["@dynamics:input_resistance"].to_list(), expected_rin, rtol=1e-5)
    npt.assert_allclose(
        df["@dynamics:holding_current"].to_list(),
        [-0.028562604715887, -0.035378993149493],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:threshold_current"].to_list(), [0.08203125, 0.1109375], rtol=1e-5
    )

    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "-v", "compute_currents",
            "--input-path", str(DATA / "sonata_v6.h5"),
            "--output-path", str(tmpdir / "sonata_currents_only_rin.h5"),
            "--morphology-path", str(DATA / "morphologies"),
            "--protocol-config-path", str(DATA / "protocol_config.yaml"),
            "--hoc-path", str(DATA / "hoc"),
            "--parallel-lib", None,
            "--only-rin",
        ],
    )
    # fmt: on
    assert response.exit_code == 0

    df = CellCollection().load_sonata(tmpdir / "sonata_currents_only_rin.h5").as_dataframe()
    npt.assert_allclose(df["@dynamics:resting_potential"].to_list(), expected_rmp, rtol=1e-5)
    npt.assert_allclose(df["@dynamics:input_resistance"].to_list(), expected_rin, rtol=1e-5)
    assert "@dynamics:holding_current" not in df.columns
    assert "@dynamics:threshold_current" not in df.columns


def test_evaluate(cli_runner, tmpdir):
    """Tetst cli evaluate."""
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "-v", "evaluate",
            "--input-path", str(DATA / "sonata_v6.h5"),
            "--output-path", str(tmpdir / "evaluation_df.csv"),
            "--morphology-path", str(DATA / "morphologies"),
            "--config-path", str(DATA / "config"),
            "--final-path", str(DATA / "final.json"),
            "--parallel-lib", None,
            "--evaluate-all",
        ],
    )
    # fmt: on

    assert response.exit_code == 0

    df = pd.read_csv(tmpdir / "evaluation_df.csv")
    # df.drop(columns=["path"]).to_csv(DATA / "evaluation_df.csv", index=None)
    expected_df = pd.read_csv(DATA / "evaluation_df.csv")

    for f, f_exp in zip(
        json.loads(df.loc[0, "features"]).values(),
        json.loads(expected_df.loc[0, "features"]).values(),
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)
    for f, f_exp in zip(
        json.loads(df.loc[1, "features"]).values(),
        json.loads(expected_df.loc[1, "features"]).values(),
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)
    for f, f_exp in zip(
        json.loads(df.loc[0, "scores"]).values(), json.loads(expected_df.loc[0, "scores"]).values()
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)
    for f, f_exp in zip(
        json.loads(df.loc[1, "scores"]).values(), json.loads(expected_df.loc[1, "scores"]).values()
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)


def test_adapt(cli_runner, tmpdir):
    """Test cli adapt."""
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "-v", "adapt",
            "--input-node-path", str(DATA / "sonata_v6.h5"),
            "--output-node-path", str(tmpdir / "sonata_v6_adapted.h5"),
            "--morphology-path", str(DATA / "morphologies"),
            "--config-path", str(DATA / "config"),
            "--final-path", str(DATA / "final.json"),
            "--local-dir", str(tmpdir / 'local'),
            "--output-hoc-path", str(tmpdir / "hoc"),
            "--parallel-lib", None,
            "--min-scale", 0.9,
            "--max-scale", 1.1,
        ],
    )
    # fmt: on
    assert response.exit_code == 0

    df = pd.read_csv(tmpdir / "local" / "adapt_df.csv")
    # df.drop(columns=["path"]).to_csv(DATA / "adapt_df.csv", index=None)
    expected_df = pd.read_csv(DATA / "adapt_df.csv")
    assert df.loc[0, "ais_scaler"] == expected_df.loc[0, "ais_scaler"]
    assert df.loc[0, "soma_scaler"] == expected_df.loc[0, "soma_scaler"]
    assert df.loc[0, "ais_model"] == expected_df.loc[0, "ais_model"]
    assert df.loc[0, "soma_model"] == expected_df.loc[0, "soma_model"]

    # retest evaluate with ais and soma scalers
    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "-v", "evaluate",
            "--input-path", str(tmpdir / "sonata_v6_adapted.h5"),
            "--output-path", str(tmpdir / "adapted_evaluation_df.csv"),
            "--morphology-path", str(DATA / "morphologies"),
            "--config-path", str(DATA / "config"),
            "--local-dir", str(tmpdir / 'local'),
            "--final-path", str(DATA / "final.json"),
            "--parallel-lib", None,
        ],
    )
    # fmt: on
    assert response.exit_code == 0

    df = pd.read_csv(tmpdir / "adapted_evaluation_df.csv")
    # df.drop(columns=["path"]).to_csv(DATA / "adapted_evaluation_df.csv", index=None)
    expected_df = pd.read_csv(DATA / "adapted_evaluation_df.csv")

    for f, f_exp in zip(
        json.loads(df.loc[0, "features"]).values(),
        json.loads(expected_df.loc[0, "features"]).values(),
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)
    for f, f_exp in zip(
        json.loads(df.loc[1, "features"]).values(),
        json.loads(expected_df.loc[1, "features"]).values(),
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)
    for f, f_exp in zip(
        json.loads(df.loc[0, "scores"]).values(), json.loads(expected_df.loc[0, "scores"]).values()
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)
    for f, f_exp in zip(
        json.loads(df.loc[1, "scores"]).values(), json.loads(expected_df.loc[1, "scores"]).values()
    ):
        npt.assert_allclose(f, f_exp, rtol=1e-1)

    # fmt: off
    response = cli_runner.invoke(
        tested.cli,
        [
            "-v", "compute_currents",
            "--input-path", str(tmpdir / "sonata_v6_adapted.h5"),
            "--output-path", str(tmpdir / "sonata_currents_adapted.h5"),
            "--morphology-path", str(DATA / "morphologies"),
            "--protocol-config-path", str(DATA / "protocol_config.yaml"),
            "--hoc-path", str(tmpdir / "hoc"),
            "--parallel-lib", None,
        ],
    )
    # fmt: on
    assert response.exit_code == 0

    df = CellCollection().load_sonata(tmpdir / "sonata_currents_adapted.h5").as_dataframe()
    npt.assert_allclose(
        df["@dynamics:resting_potential"].to_list(),
        [-72.841806, -71.32893],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:input_resistance"].to_list(),
        [105.342194, 1863.809101],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:holding_current"].to_list(),
        [-0.06431251604510635, -0.08120532523605561],
        rtol=1e-5,
    )
    npt.assert_allclose(
        df["@dynamics:threshold_current"].to_list(),
        [0.05625, 0.06875],
        rtol=1e-5,
    )
