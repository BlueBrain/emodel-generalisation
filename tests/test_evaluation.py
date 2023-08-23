"""
Copyright (c) 2022 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
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

from numpy.testing import assert_allclose

from emodel_generalisation.model import evaluation


def test_feature_evaluation(morphs_combos_df, access_point):
    """ """
    df = evaluation.feature_evaluation(morphs_combos_df, access_point)

    expected_scores = {
        "IDRest_150.soma.v.mean_frequency": 2.3567374199026414,
        "IDRest_150.soma.v.AP_amplitude": 0.624398677191681,
        "IDRest_200.soma.v.mean_frequency": 2.2226560510494124,
        "IDRest_200.soma.v.AP_amplitude": 0.5340149936839824,
        "IDRest_280.soma.v.mean_frequency": 2.5353330326086994,
        "IDRest_280.soma.v.AP_amplitude": 0.4711507560288358,
        "RMPProtocol.soma.v.voltage_base": 0.33131119446020707,
        "RMPProtocol.soma.v.Spikecount": 0.0,
        "SearchHoldingCurrent.soma.v.bpo_holding_current": 0.18281250000000002,
        "SearchHoldingCurrent.soma.v.voltage_base": 0.043970045446585004,
        "RinProtocol.soma.v.ohmic_input_resistance_vb_ssse": 0.17312024013731503,
        "SearchThresholdCurrent.soma.v.bpo_threshold_current": 0.10935385147499851,
    }

    scores = json.loads(df.loc[0, "scores"])
    for feature, score in expected_scores.items():
        assert_allclose(score, scores[feature], rtol=1e-1)

    expected_features = {
        "IDRest_150.soma.v.mean_frequency": 19.713474839805283,
        "IDRest_150.soma.v.AP_amplitude": 68.75120264561664,
        "IDRest_200.soma.v.mean_frequency": 24.445312102098825,
        "IDRest_200.soma.v.AP_amplitude": 68.93197001263204,
        "IDRest_280.soma.v.mean_frequency": 30.0706660652174,
        "IDRest_280.soma.v.AP_amplitude": 69.05769848794233,
        "RMPProtocol.soma.v.voltage_base": -75.34344402769896,
        "RMPProtocol.soma.v.Spikecount": 0.0,
        "SearchHoldingCurrent.soma.v.bpo_holding_current": -0.00859375,
        "SearchHoldingCurrent.soma.v.voltage_base": -82.95602995455341,
        "RinProtocol.soma.v.ohmic_input_resistance_vb_ssse": 517.3120240137315,
        "SearchThresholdCurrent.soma.v.bpo_threshold_current": 0.015467692573749926,
    }

    features = json.loads(df.loc[0, "features"])
    for feature, value in expected_features.items():
        assert_allclose(value, features[feature], rtol=1e-1)


def test_evaluate_rin_no_soma(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_rin_no_soma(morphs_combos_df, access_point)
    assert_allclose(df.loc[0, "rin_no_soma"], 737.355812849696, rtol=1e-3)


def test_evaluate_soma_rin(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_soma_rin(morphs_combos_df, access_point)
    assert_allclose(df.loc[0, "rin_soma"], 848.5805718798488, rtol=1e-5)


def test_evaluate_ais_rin(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_ais_rin(morphs_combos_df, access_point)
    assert_allclose(df.loc[0, "rin_ais"], 131668.9953893715, rtol=1e-5)


def test_evaluate_somadend_rin(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_somadend_rin(morphs_combos_df, access_point)
    assert_allclose(df.loc[0, "rin_no_ais"], 539.1371042337126, rtol=1e-5)


def test_evaluate_rho(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_rho(morphs_combos_df, access_point)
    assert_allclose(df.loc[0, "rho"], 1.150842747082164, rtol=1e-3)


def test_evaluate_rho_axon(morphs_combos_df, access_point):
    """ """
    df = evaluation.evaluate_rho_axon(morphs_combos_df, access_point)
    assert_allclose(df.loc[0, "rho_axon"], 244.2217237122562, rtol=1e-5)
