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

"""Test evaluation module."""
import json

from numpy.testing import assert_almost_equal

from emodel_generalisation.model import evaluation


def test_feature_evaluation(morphs_combos_df, access_point):
    """ """
    df = evaluation.feature_evaluation(morphs_combos_df, access_point)

    expected_scores = {
        "IDRest_150.soma.v.mean_frequency": 0.38832870321843593,
        "IDRest_150.soma.v.AP_amplitude": 0.12185120679916672,
        "IDRest_200.soma.v.mean_frequency": 0.15417421177041213,
        "IDRest_200.soma.v.AP_amplitude": 0.15734097154168272,
        "IDRest_280.soma.v.mean_frequency": 0.36739807664764434,
        "IDRest_280.soma.v.AP_amplitude": 0.17813046992506038,
        "RMPProtocol.soma.v.voltage_base": 0.13535686096089705,
        "RMPProtocol.soma.v.Spikecount": 0.0,
        "SearchHoldingCurrent.soma.v.bpo_holding_current": 0.185546875,
        "SearchHoldingCurrent.soma.v.voltage_base": 0.026415106600779836,
        "RinProtocol.soma.v.ohmic_input_resistance_vb_ssse": 0.28658363664524017,
        "SearchThresholdCurrent.soma.v.bpo_threshold_current": 0.011158751197907664,
    }

    scores = json.loads(df.loc[0, "scores"])
    for feature, score in expected_scores.items():
        assert_almost_equal(score, scores[feature], decimal=1)

    expected_features = {
        "IDRest_150.soma.v.mean_frequency": 15.776657406436872,
        "IDRest_150.soma.v.AP_amplitude": 70.24370241359833,
        "IDRest_200.soma.v.mean_frequency": 20.308348423540824,
        "IDRest_200.soma.v.AP_amplitude": 70.31468194308337,
        "IDRest_280.soma.v.mean_frequency": 25.73479615329529,
        "IDRest_280.soma.v.AP_amplitude": 70.35626093985013,
        "RMPProtocol.soma.v.voltage_base": -76.32321569519551,
        "RMPProtocol.soma.v.Spikecount": 0.0,
        "SearchHoldingCurrent.soma.v.bpo_holding_current": -0.0072265625,
        "SearchHoldingCurrent.soma.v.voltage_base": -83.02641510660078,
        "RinProtocol.soma.v.ohmic_input_resistance_vb_ssse": 528.658363664524,
        "SearchThresholdCurrent.soma.v.bpo_threshold_current": 0.010557937559895383,
    }

    features = json.loads(df.loc[0, "features"])
    for feature, value in expected_features.items():
        assert_almost_equal(value, features[feature], decimal=1)


def test_evaluate_rin_no_soma(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_rin_no_soma(morphs_combos_df, access_point)
    assert_almost_equal(df.loc[0, "rin_no_soma"], 737.355812849696, decimal=0)


def test_evaluate_soma_rin(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_soma_rin(morphs_combos_df, access_point)
    assert_almost_equal(df.loc[0, "rin_soma"], 848.5805718798488, decimal=3)


def test_evaluate_ais_rin(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_ais_rin(morphs_combos_df, access_point)
    assert_almost_equal(df.loc[0, "rin_ais"], 131668.9953893715, decimal=3)


def test_evaluate_somadend_rin(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_somadend_rin(morphs_combos_df, access_point)
    assert_almost_equal(df.loc[0, "rin_no_ais"], 539.1371042337126, decimal=3)


def test_evaluate_rho(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_rho(morphs_combos_df, access_point)
    assert_almost_equal(df.loc[0, "rho"], 1.1510177066149903, decimal=3)


def test_evaluate_rho_axon(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_rho_axon(morphs_combos_df, access_point)
    assert_almost_equal(df.loc[0, "rho_axon"], 244.2217237122562, decimal=3)
