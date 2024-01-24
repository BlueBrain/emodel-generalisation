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

"""Test AccessPoint module."""
from pathlib import Path

from emodel_generalisation.model.access_point import AccessPoint

DATA = Path(__file__).parent / "data"
EMODEL = "generic_model"


def test___init__(access_point):
    """ """
    assert access_point.emodel_dir == DATA
    assert access_point.recipes_path == DATA / "config" / "recipes.json"
    assert access_point.legacy_dir_structure is False
    assert access_point.with_seeds is False
    assert EMODEL in access_point.final


def test_get_recipes(access_point):
    """ """
    assert access_point.get_recipes(EMODEL) == {
        "morph_path": ".",
        "morphology": "simple.asc",
        "params": "config/parameters.json",
        "features": "config/features.json",
        "morph_modifiers": [],
        "pipeline_settings": {
            "efel_settings": {
                "strict_stiminterval": True,
                "Threshold": -30.0,
                "interp_step": 0.025,
            },
            "name_rmp_protocol": "IV_0",
            "name_Rin_protocol": "IV_-20",
        },
    }


def test_get_settings(access_point):
    """ """
    access_point.settings = {"test": 0}
    assert access_point.get_settings(EMODEL) == {
        "efel_settings": {"strict_stiminterval": True, "Threshold": -30.0, "interp_step": 0.025},
        "name_rmp_protocol": "IV_0",
        "name_Rin_protocol": "IV_-20",
        "test": 0,
        "morph_modifiers": [],
    }


def test_get_calculator_configuration(access_point):
    """ """
    config = access_point.get_calculator_configuration(EMODEL)
    assert config.as_dict() == {
        "efeatures": [
            {
                "efel_feature_name": "mean_frequency",
                "protocol_name": "IDRest_150",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 1.0,
                "mean": 15.0,
                "original_std": 2.0,
                "sample_size": None,
                "efeature_name": "mean_frequency",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "AP_amplitude",
                "protocol_name": "IDRest_150",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": 70.0,
                "original_std": 2.0,
                "sample_size": None,
                "efeature_name": "AP_amplitude",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "mean_frequency",
                "protocol_name": "IDRest_200",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 1.0,
                "mean": 20.0,
                "original_std": 2.0,
                "sample_size": None,
                "efeature_name": "mean_frequency",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "AP_amplitude",
                "protocol_name": "IDRest_200",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": 70.0,
                "original_std": 2.0,
                "sample_size": None,
                "efeature_name": "AP_amplitude",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "mean_frequency",
                "protocol_name": "IDRest_280",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 1.0,
                "mean": 25.0,
                "original_std": 2.0,
                "sample_size": None,
                "efeature_name": "mean_frequency",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "AP_amplitude",
                "protocol_name": "IDRest_280",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": 70.0,
                "original_std": 2.0,
                "sample_size": None,
                "efeature_name": "AP_amplitude",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "steady_state_voltage_stimend",
                "protocol_name": "RMPProtocol",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": -77,
                "original_std": 5.0,
                "sample_size": None,
                "efeature_name": "voltage_base",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "Spikecount",
                "protocol_name": "RMPProtocol",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": 0,
                "original_std": 0.001,
                "sample_size": None,
                "efeature_name": "Spikecount",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "bpo_holding_current",
                "protocol_name": "SearchHoldingCurrent",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": -0.1,
                "original_std": 0.5,
                "sample_size": None,
                "efeature_name": None,
                "efel_settings": {"strict_stiminterval": True},
            },
            {
                "efel_feature_name": "steady_state_voltage_stimend",
                "protocol_name": "SearchHoldingCurrent",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": -83,
                "original_std": 1.0,
                "sample_size": None,
                "efeature_name": "voltage_base",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "ohmic_input_resistance_vb_ssse",
                "protocol_name": "RinProtocol",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": 500.0,
                "original_std": 100.0,
                "sample_size": None,
                "efeature_name": "ohmic_input_resistance_vb_ssse",
                "efel_settings": {
                    "strict_stiminterval": True,
                    "Threshold": -30.0,
                    "interp_step": 0.025,
                },
            },
            {
                "efel_feature_name": "bpo_threshold_current",
                "protocol_name": "SearchThresholdCurrent",
                "recording_name": "soma.v",
                "threshold_efeature_std": None,
                "default_std_value": 0.001,
                "mean": 0.01,
                "original_std": 0.05,
                "sample_size": None,
                "efeature_name": None,
                "efel_settings": {"strict_stiminterval": True},
            },
        ],
        "protocols": [
            {
                "name": "IDRest_150",
                "stimuli": [
                    {
                        "delay": 200,
                        "thresh_perc": 150.0,
                        "duration": 1500,
                        "totduration": 2000,
                        "holding_current": -0.3,
                    }
                ],
                "recordings_from_config": [
                    {
                        "type": "CompRecording",
                        "name": "IDRest_150.soma.v",
                        "location": "soma",
                        "variable": "v",
                    }
                ],
                "validation": False,
                "protocol_type": "ThresholdBasedProtocol",
                "stochasticity": False,
            },
            {
                "name": "IDRest_200",
                "stimuli": [
                    {
                        "delay": 200,
                        "thresh_perc": 200,
                        "duration": 1500,
                        "totduration": 2000,
                        "holding_current": -0.3,
                    }
                ],
                "recordings_from_config": [
                    {
                        "type": "CompRecording",
                        "name": "IDRest_200.soma.v",
                        "location": "soma",
                        "variable": "v",
                    }
                ],
                "validation": False,
                "protocol_type": "ThresholdBasedProtocol",
                "stochasticity": False,
            },
            {
                "name": "IDRest_280",
                "stimuli": [
                    {
                        "delay": 200,
                        "thresh_perc": 280,
                        "duration": 1500,
                        "totduration": 2000,
                        "holding_current": -0.3,
                    }
                ],
                "recordings_from_config": [
                    {
                        "type": "CompRecording",
                        "name": "IDRest_280.soma.v",
                        "location": "soma",
                        "variable": "v",
                    }
                ],
                "validation": False,
                "protocol_type": "ThresholdBasedProtocol",
                "stochasticity": False,
            },
        ],
    }


def test_get_mechanisms_directory(access_point):
    """ """
    assert access_point.get_mechanisms_directory() == DATA / "mechanisms"


def test_get_available_mechanisms(access_point):
    """ """
    assert sorted([m.name for m in access_point.get_available_mechanisms()]) == [
        "CaDynamics_DC0",
        "Ca_HVA",
        "Ca_HVA2",
        "Ca_LVAst",
        "Ih",
        "K_Pst",
        "K_Tst",
        "KdShu2007",
        "NaTg",
        "NaTg2",
        "Nap_Et2",
        "SK_E2",
        "SKv3_1",
        "StochKv3",
        "TTXDynamicsSwitch",
    ]


def test_get_json(access_point):
    """ """
    # pylint: disable=line-too-long
    assert access_point.get_json(EMODEL, "params") == {
        "mechanisms": {
            "all": {"mech": ["pas"]},
            "somatic": {"mech": ["SKv3_1", "K_Pst", "NaTg"]},
            "somadend": {"mech": ["Ih"]},
            "axonal": {"mech": ["SKv3_1", "K_Pst", "NaTg"]},
        },
        "distributions": {
            "exp": {
                "fun": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}",
                "__comment": "distribution based on Kole et al. 2006",
            }
        },
        "parameters": {
            "global": [{"name": "v_init", "val": -80}, {"name": "celsius", "val": 34}],
            "myelinated": [{"name": "cm", "val": 0.02}],
            "all": [{"name": "g_pas", "val": [0, 1e-05]}],
            "somadend": [
                {
                    "name": "gIhbar_Ih",
                    "val": [8e-05, 0.00015],
                    "dist": "exp",
                    "__comment": "distribution starts in soma (uniform) and spreads exponentially to dendrites",  # noqa
                }
            ],
            "somatic": [
                {"name": "cm", "val": 1.0},
                {"name": "ena", "val": 50},
                {"name": "ek", "val": -90},
                {"name": "e_pas", "val": -90},
                {"name": "gSKv3_1bar_SKv3_1", "val": [0, 0.2]},
                {"name": "gK_Pstbar_K_Pst", "val": [0, 1.0]},
                {"name": "gNaTgbar_NaTg", "val": [0, 0.01]},
                {"name": "vshiftm_NaTg", "val": 13},
                {"name": "vshifth_NaTg", "val": 15},
            ],
            "axonal": [
                {"name": "cm", "val": 1.0},
                {"name": "ena", "val": 50},
                {"name": "ek", "val": -90},
                {"name": "e_pas", "val": -90},
                {"name": "gSKv3_1bar_SKv3_1", "val": [1.0, 2.0]},
                {"name": "gK_Pstbar_K_Pst", "val": [0, 2.0]},
                {"name": "gNaTgbar_NaTg", "val": [0.45, 0.9]},
                {"name": "slopem_NaTg", "val": 9},
                {"name": "vshifth_NaTg", "val": 10},
            ],
        },
    }


def test_get_morphologies(access_point):
    """ """
    assert access_point.get_morphologies(EMODEL) == {
        "name": "simple",
        "path": str(DATA / "simple.asc"),
    }


def test_get_configuration(access_point):
    """ """
    config = access_point.get_configuration(EMODEL).as_dict()

    assert config == {
        "mechanisms": [
            {"name": "pas", "stochastic": False, "location": "all", "version": None},
            {"name": "Ih", "stochastic": False, "location": "somadend", "version": None},
            {"name": "SKv3_1", "stochastic": False, "location": "somatic", "version": None},
            {"name": "K_Pst", "stochastic": False, "location": "somatic", "version": None},
            {"name": "NaTg", "stochastic": False, "location": "somatic", "version": None},
            {"name": "SKv3_1", "stochastic": False, "location": "axonal", "version": None},
            {"name": "K_Pst", "stochastic": False, "location": "axonal", "version": None},
            {"name": "NaTg", "stochastic": False, "location": "axonal", "version": None},
        ],
        "distributions": [
            {"name": "uniform", "function": None, "soma_ref_location": 0.5},
            {
                "name": "exp",
                "function": "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}",
                "soma_ref_location": 0.5,
            },
        ],
        "parameters": [
            {"name": "v_init", "value": -80, "location": "global"},
            {"name": "celsius", "value": 34, "location": "global"},
            {"name": "cm", "value": 0.02, "location": "myelinated"},
            {"name": "g_pas", "value": [0, 1e-05], "location": "all", "mechanism": "pas"},
            {
                "name": "gIhbar_Ih",
                "value": [8e-05, 0.00015],
                "location": "somadend",
                "distribution": "exp",
                "mechanism": "Ih",
            },
            {"name": "cm", "value": 1.0, "location": "somatic"},
            {"name": "ena", "value": 50, "location": "somatic"},
            {"name": "ek", "value": -90, "location": "somatic"},
            {"name": "e_pas", "value": -90, "location": "somatic", "mechanism": "pas"},
            {
                "name": "gSKv3_1bar_SKv3_1",
                "value": [0, 0.2],
                "location": "somatic",
                "mechanism": "SKv3_1",
            },
            {
                "name": "gK_Pstbar_K_Pst",
                "value": [0, 1.0],
                "location": "somatic",
                "mechanism": "K_Pst",
            },
            {
                "name": "gNaTgbar_NaTg",
                "value": [0, 0.01],
                "location": "somatic",
                "mechanism": "NaTg",
            },
            {"name": "vshiftm_NaTg", "value": 13, "location": "somatic", "mechanism": "NaTg"},
            {"name": "vshifth_NaTg", "value": 15, "location": "somatic", "mechanism": "NaTg"},
            {"name": "cm", "value": 1.0, "location": "axonal"},
            {"name": "ena", "value": 50, "location": "axonal"},
            {"name": "ek", "value": -90, "location": "axonal"},
            {"name": "e_pas", "value": -90, "location": "axonal", "mechanism": "pas"},
            {
                "name": "gSKv3_1bar_SKv3_1",
                "value": [1.0, 2.0],
                "location": "axonal",
                "mechanism": "SKv3_1",
            },
            {
                "name": "gK_Pstbar_K_Pst",
                "value": [0, 2.0],
                "location": "axonal",
                "mechanism": "K_Pst",
            },
            {
                "name": "gNaTgbar_NaTg",
                "value": [0.45, 0.9],
                "location": "axonal",
                "mechanism": "NaTg",
            },
            {"name": "slopem_NaTg", "value": 9, "location": "axonal", "mechanism": "NaTg"},
            {"name": "vshifth_NaTg", "value": 10, "location": "axonal", "mechanism": "NaTg"},
        ],
        "morphology": {
            "name": "simple",
            "format": "asc",
            "path": str(DATA / "simple.asc"),
            "seclist_names": None,
            "secarray_names": None,
            "section_index": None,
        },
        "morph_modifiers": None,
    }


def test_load_nexus_recipe(tmpdir):
    """Test loading a nexus recipe by converting it to a config folder."""
    access_point = AccessPoint(
        nexus_config=DATA / "nexus_recipe.json",
        emodel_dir=tmpdir / "config",
    )
    assert access_point.emodels == ["c4de21"]

    access_point = AccessPoint(
        nexus_config=DATA / "nexus_recipe.json",
        emodel_dir=tmpdir / "config",
    )
    assert access_point.emodels == ["c4de21"]

    assert (tmpdir / "config" / "recipes.json").exists()
    assert (tmpdir / "config" / "final.json").exists()
    assert (tmpdir / "config" / "parameters" / "c4de21.json").exists()
    assert (tmpdir / "config" / "features" / "c4de21.json").exists()
