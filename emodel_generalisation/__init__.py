"""Emodel generalisation package."""

# Copyright (c) 2022 EPFL-BBP, All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode
# or send a letter to Creative Commons, 171
# Second Street, Suite 300, San Francisco, California, 94105, USA.

# pylint: disable=line-too-long
import logging
import os
from pathlib import Path

import neuron

os.environ["NEURON_MODULE_OPTIONS"] = "-nogui"
logger = logging.getLogger(__name__)


PARAM_LABELS = {
    # passive
    "g_pas.all": "all: ḡpas",
    "g_pas.somatic": "soma: ḡpas",
    "e_pas.all": "all: Epas",
    "e_pas.somatic": "soma: Epas",
    "gIhbar_Ih.somadend": "soma, dendrites: ḡIh",
    "gIhbar_Ih.somatic": "soma: ḡIh",
    # soma
    "decay_CaDynamics_DC0.somatic": "soma: Ca2+ decay time constant",
    "gamma_CaDynamics_DC0.somatic": "soma: % of free Ca2+",
    "gNaTgbar_NaTg.somatic": "soma: ḡNaT",
    "gNaTgbar_NaTg2.somatic": "soma: ḡNaT",
    "gK_Tstbar_K_Tst.somatic": "soma: ḡKT",
    "gK_Pstbar_K_Pst.somatic": "soma: ḡKP",
    "gSKv3_1bar_SKv3_1.somatic": "soma: ḡKv3.1",
    "gSK_E2bar_SK_E2.somatic": "soma: ḡSK",
    "gSK_E2bar_SK_E2.alldend": "dendrites: ḡSK",
    "gCa_HVAbar_Ca_HVA2.somatic": "soma: ḡCa_HVA",
    "gCa_HVAbar_Ca_HVA2.somadend": "soma, dendrites: ḡCa_HVA",
    "gCa_LVAstbar_Ca_LVAst.somatic": "soma: ḡCa_LVA",
    "gCa_LVAstbar_Ca_LVAst.somadend": "soma, dendrites: ḡCa_LVA",
    # axon
    "decay_CaDynamics_DC0.axonal": "axon: Ca2+ decay time constant",
    "gamma_CaDynamics_DC0.axonal": "axon: % of free Ca2+",
    "gNaTgbar_NaTg.axonal": "axon: ḡNaT",
    "gNap_Et2bar_Nap_Et2.axonal": "axon: ḡNaP",
    "gK_Tstbar_K_Tst.axonal": "axon: ḡKT",
    "gK_Pstbar_K_Pst.axonal": "axon: ḡKP",
    "gSKv3_1bar_SKv3_1.axonal": "axon: ḡKv3.1",
    "gSK_E2bar_SK_E2.axonal": "axon: ḡSK",
    "gCa_HVAbar_Ca_HVA2.axonal": "axon: ḡCa_HVA",
    "gCa_LVAstbar_Ca_LVAst.axonal": "axon: ḡCa_LVA",
    # basal
    "decay_CaDynamics_DC0.basal": "basal: Ca2+ decay time constant",
    "gamma_CaDynamics_DC0.basal": "basal: % of free Ca2+",
    "gNaTgbar_NaTg.basal": "basal: ḡNaT",
    "gSK_E2bar_SK_E2.basal": "basal: ḡSK",
    "gCa_HVAbar_Ca_HVA2.basal": "basal: ḡCa_HVA",
    "gCa_LVAstbar_Ca_LVAst.basal": "basal: ḡCa_LVA",
    # apical
    "constant.distribution_decay": "apical: exp decay constant NaT",
    "decay_CaDynamics_DC0.apical": "apical: Ca2+ decay time constant",
    "gamma_CaDynamics_DC0.apical": "apical: % of free Ca2+",
    "gNaTgbar_NaTg.apical": "apical: ḡNaT",
    "gSKv3_1bar_SKv3_1.apical": "apical: ḡKv3.1",
    "gSK_E2bar_SK_E2.apical": "apical: ḡSK",
    "gCa_HVAbar_Ca_HVA2.apical": "apical: ḡCa_HVA",
    "gCa_LVAstbar_Ca_LVAst.apical": "apical: ḡCa_LVA",
}

PARAM_LABELS_2 = {
    "g_pas.all": "all: ḡpas",
    "e_pas.all": "all: Epas",
    "gIhbar_Ih.somadend": "soma, dendrites: ḡIh",
    "constant.distribution_decay": "apical: exp decay constant NaT",
    "decay_CaDynamics_DC0.somatic": "soma: Ca2+ decay time constant",
    "decay_CaDynamics_DC0.axonal": "axon: Ca2+ decay time constant",
    "decay_CaDynamics_DC0.basal": "basal: Ca2+ decay time constant",
    "decay_CaDynamics_DC0.apical": "apical: Ca2+ decay time constant",
    "gamma_CaDynamics_DC0.somatic": "soma: % of free Ca2+",
    "gamma_CaDynamics_DC0.axonal": "axon: % of free Ca2+",
    "gamma_CaDynamics_DC0.basal": "basal: % of free Ca2+",
    "gamma_CaDynamics_DC0.apical": "apical: % of free Ca2+",
    "gNaTgbar_NaTg.somatic": "soma: ḡNaT",
    "gNaTgbar_NaTg.axonal": "axon: ḡNaT",
    "gNaTgbar_NaTg.basal": "basal: ḡNaT",
    "gNaTgbar_NaTg.apical": "apical: ḡNaT",
    "gNap_Et2bar_Nap_Et2.axonal": "axon: ḡNaP",
    "gK_Tstbar_K_Tst.somatic": "soma: ḡKT",
    "gK_Tstbar_K_Tst.axonal": "axon: ḡKT",
    "gK_Pstbar_K_Pst.somatic": "soma: ḡKP",
    "gK_Pstbar_K_Pst.axonal": "axon: ḡKP",
    "gSKv3_1bar_SKv3_1.somatic": "soma: ḡKv3.1",
    "gSKv3_1bar_SKv3_1.axonal": "axon: ḡKv3.1",
    "gSKv3_1bar_SKv3_1.apical": "apical: ḡKv3.1",
    "gSK_E2bar_SK_E2.somatic": "soma: ḡSK",
    "gSK_E2bar_SK_E2.axonal": "axon: ḡSK",
    "gSK_E2bar_SK_E2.basal": "basal: ḡSK",
    "gSK_E2bar_SK_E2.apical": "apical: ḡSK",
    "gSK_E2bar_SK_E2.alldend": "dendrites: ḡSK",
    "gCa_HVAbar_Ca_HVA2.somatic": "soma: ḡCa_HVA",
    "gCa_HVAbar_Ca_HVA2.somadend": "soma, dendrites: ḡCa_HVA",
    "gCa_HVAbar_Ca_HVA2.axonal": "axon: ḡCa_HVA",
    "gCa_HVAbar_Ca_HVA2.basal": "basal: ḡCa_HVA",
    "gCa_HVAbar_Ca_HVA2.apical": "apical: ḡCa_HVA",
    "gCa_LVAstbar_Ca_LVAst.somatic": "soma: ḡCa_LVA",
    "gCa_LVAstbar_Ca_LVAst.somadend": "soma, dendrites: ḡCa_LVA",
    "gCa_LVAstbar_Ca_LVAst.axonal": "axon: ḡCa_LVA",
    "gCa_LVAstbar_Ca_LVAst.basal": "basal: ḡCa_LVA",
    "gCa_LVAstbar_Ca_LVAst.apical": "apical: ḡCa_LVA",
}

FEATURE_LABELS_LONG = {
    "SearchHoldingCurrent.soma.v.bpo_holding_current": "Holding current for -80 mV RMP",
    "SearchHoldingCurrent.soma.v.steady_state_voltage_stimend": "Holding: Voltage after stim",
    "SearchThresholdCurrent.soma.v.bpo_threshold_current": "Rheobase",
    "APWaveform_320.soma.v.AHP_depth": "APWaveform: Mean AHP depth relative to RMP",
    "APWaveform_320.soma.v.AP1_amp": "APWaverform: 1st AP amplitude",
    "APWaveform_320.soma.v.AP2_amp": "APWaverform: 2nd AP amplitude",
    "APWaveform_320.soma.v.AP_amplitude": "APWaverform: Mean AP amplitude",
    "APWaveform_320.soma.v.AP_duration_half_width": "APWaverform: Mean AP half-width",
    "IV_-100.soma.v.voltage_deflection": "Sag: Voltage hyperpolarization end stim",
    "IV_-100.soma.v.voltage_deflection_begin": "Sag: Voltage hyperpolarization begin stim",
    "RMPProtocol.soma.v.steady_state_voltage_stimend": "RMP",
    "RMPProtocol.soma.v.Spikecount": "RMP: Nb. of APs",
    "RinProtocol.soma.v.ohmic_input_resistance_vb_ssse": "Rm",
    "SpikeRec_600.soma.v.Spikecount": "SpikeRec: Spike count",
    "SpikeRec_600.soma.v.decay_time_constant_after_stim": "SpikeRec: Decay time constant",
    "SpikeRec_600.soma.v.voltage_after_stim": "SpikeRec: Voltage value after stim",
    "Step_150.soma.v.AHP_depth": "Step 150% of rheobase: Mean AHP depth relative to RMP",
    "Step_150.soma.v.AP_amplitude": "Step 150% of rheobase: Mean AP amplitude",
    "Step_150.soma.v.APlast_amp": "Step 150% of rheobase: Last AP amplitude",
    "Step_150.soma.v.inv_fifth_ISI": "Step 150% of rheobase: Inverse of 5th ISI",
    "Step_150.soma.v.inv_first_ISI": "Step 150% of rheobase: Inverse of 1st ISI",
    "Step_150.soma.v.inv_fourth_ISI": "Step 150% of rheobase: Inverse of 4th ISI",
    "Step_150.soma.v.inv_second_ISI": "Step 150% of rheobase: Inverse of 2nd ISI",
    "Step_150.soma.v.inv_third_ISI": "Step 150% of rheobase: Inverse of 3rd ISI",
    "Step_150.soma.v.inv_time_to_first_spike": "Step 150% of rheobase: Inverse of 1st AP time from stim start",  # noqa
    "Step_150.soma.v.mean_frequency": "Step 150% of rheobase: Mean firing frequency",
    "Step_150.soma.v.time_to_last_spike": "Step 150% of rheobase: Inverse of last AP time from stim start",  # noqa
    "Step_150.soma.v.voltage_after_stim": "Step 150% of rheobase: Voltage value after stim",
    "Step_150.soma.v.voltage_base": "Step 150% of rheobase: RMP",
    "Step_200.soma.v.AHP_depth": "Step 200% of rheobase: Mean AHP depth relative to RMP",
    "Step_200.soma.v.AP_amplitude": "Step 200% of rheobase: Mean AP amplitude",
    "Step_200.soma.v.APlast_amp": "Step 200% of rheobase: Last AP amplitude",
    "Step_200.soma.v.inv_fifth_ISI": "Step 200% of rheobase: Inverse of 5th ISI",
    "Step_200.soma.v.inv_first_ISI": "Step 200% of rheobase: Inverse of 1st ISI",
    "Step_200.soma.v.inv_fourth_ISI": "Step 200% of rheobase: Inverse of 4th ISI",
    "Step_200.soma.v.inv_second_ISI": "Step 200% of rheobase: Inverse of 2nd ISI",
    "Step_200.soma.v.inv_third_ISI": "Step 200% of rheobase: Inverse of 3rd ISI",
    "Step_200.soma.v.inv_time_to_first_spike": "Step 200% of rheobase: Inverse of 1st AP time from stim start",  # noqa
    "Step_200.soma.v.mean_frequency": "Step 200% of rheobase: Mean firing frequency",
    "Step_200.soma.v.time_to_last_spike": "Step 200% of rheobase: Inverse of last AP time from stim start",  # noqa
    "Step_200.soma.v.voltage_after_stim": "Step 200% of rheobase: Voltage value after stim",
    "Step_200.soma.v.voltage_base": "Step 200% of rheobase: RMP",
    "Step_280.soma.v.AHP_depth": "Step 280% of rheobase: Mean AHP depth relative to RMP",
    "Step_280.soma.v.AP_amplitude": "Step 280% of rheobase: Mean AP amplitude",
    "Step_280.soma.v.APlast_amp": "Step 280% of rheobase: Last AP amplitude",
    "Step_280.soma.v.inv_fifth_ISI": "Step 280% of rheobase: Inverse of 5th ISI",
    "Step_280.soma.v.inv_first_ISI": "Step 280% of rheobase: Inverse of 1st ISI",
    "Step_280.soma.v.inv_fourth_ISI": "Step 280% of rheobase: Inverse of 4th ISI",
    "Step_280.soma.v.inv_second_ISI": "Step 280% of rheobase: Inverse of 2nd ISI",
    "Step_280.soma.v.inv_third_ISI": "Step 280% of rheobase: Inverse of 3rd ISI",
    "Step_280.soma.v.inv_time_to_first_spike": "Step 280% of rheobase: Inverse of 1st AP time from stim start",  # noqa
    "Step_280.soma.v.mean_frequency": "Step 280% of rheobase: Mean firing frequency",
    "Step_280.soma.v.time_to_last_spike": "Step 280% of rheobase: Inverse of last AP time from stim start",  # noqa
    "Step_280.soma.v.voltage_after_stim": "Step 280% of rheobase: Voltage value after stim",
    "Step_280.soma.v.voltage_base": "Step 280% of rheobase: RMP",
    "bAP.ca_ais.cai.maximum_voltage_from_voltagebase": "bAP: Max [Ca] in AIS",
    "bAP.ca_prox_apic.cai.maximum_voltage_from_voltagebase": "bAP: Max [Ca] in apical dendrite (50 µm from the soma)",  # noqa
    "bAP.ca_prox_basal.cai.maximum_voltage_from_voltagebase": "bAP: Max [Ca] in basal dendrite (50 µm from the soma)",  # noqa
    "bAP.ca_soma.cai.maximum_voltage_from_voltagebase": "bAP: Max [Ca] in soma",
    "bAP.dend1.v.maximum_voltage_from_voltagebase": "bAP: AP amplitude in apical dendrite (208 µm from the soma)",  # noqa
    "bAP.dend2.v.maximum_voltage_from_voltagebase": "bAP: AP amplitude in apical dendrite (402 µm from the soma)",  # noqa
    "bAP.soma.v.Spikecount": "bAP: Spike count",
}
ALL_LABELS = PARAM_LABELS.copy()
ALL_LABELS.update(FEATURE_LABELS_LONG)


def load_mechanisms():
    """Load mechanisms if present in TMPDIR."""
    _MOD_PATH = os.environ.get("EMODEL_GENERALISATION_MOD_LIBRARY_PATH", None)
    if _MOD_PATH is not None:
        try:
            if (Path(_MOD_PATH) / "x86_64").exists():
                if not neuron.load_mechanisms(_MOD_PATH):
                    raise Exception("Could not load mod files")
                logging.info("Mechanisms loaded from %s", _MOD_PATH)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.debug("Could not load mod files from %s because of %s", _MOD_PATH, exc)


_TMPDIR = os.environ.get("TMPDIR", False)
if _TMPDIR:
    os.environ["DASK_TEMPORARY_DIRECTORY"] = _TMPDIR

os.environ["NEURON_MODULE_OPTIONS"] = "-nogui"

load_mechanisms()
