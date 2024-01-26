"""Evaluation module (adapted from bluepyemodel)."""

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

import importlib
import json
import logging
import multiprocessing
import os
import pickle
import sys
import traceback
from copy import deepcopy
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
from bluepyopt import ephys
from bluepyopt.ephys.evaluators import CellEvaluator
from bluepyopt.ephys.locations import NrnSeclistCompLocation
from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation
from bluepyopt.ephys.locations import NrnTrunkSomaDistanceCompLocation
from bluepyopt.ephys.morphologies import NrnFileMorphology
from bluepyopt.ephys.objectives import SingletonObjective
from bluepyopt.ephys.objectivescalculators import ObjectivesCalculator
from bluepyopt.ephys.simulators import NrnSimulator

from emodel_generalisation.model import bpopt
from emodel_generalisation.model import modifiers
from emodel_generalisation.model.ecodes import eCodes
from emodel_generalisation.parallel import evaluate
from emodel_generalisation.parallel.parallel import NestedPool

# pylint: disable=too-many-lines

logger = logging.getLogger(__name__)
protocol_type_to_class = {
    "Protocol": bpopt.BPEMProtocol,
    "ThresholdBasedProtocol": bpopt.ThresholdBasedProtocol,
    "ReboundBurst": bpopt.ReboundBurst,
}

soma_loc = NrnSeclistCompLocation(name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5)
ais_loc = NrnSeclistCompLocation(name="soma", seclist_name="axonal", sec_index=0, comp_x=0.5)
multiloc_map = {
    "alldend": ["apical", "basal"],
    "somadend": ["apical", "basal", "somatic"],
    "allnoaxon": ["apical", "basal", "somatic"],
    "somaxon": ["axonal", "somatic"],
    "allact": ["apical", "basal", "somatic", "axonal"],
}


seclist_to_sec = {
    "somatic": "soma",
    "apical": "apic",
    "axonal": "axon",
    "myelinated": "myelin",
}


PRE_PROTOCOLS = ["SearchHoldingCurrent", "SearchThresholdCurrent", "RMPProtocol", "RinProtocol"]
LEGACY_PRE_PROTOCOLS = ["RMP", "Rin", "RinHoldcurrent", "Main", "ThresholdDetection"]


def _define_distributions(distributions_definition):
    """Create a list of ParameterScaler from a the definition of channel distributions

    Args:
        distributions_definition (list): definitions of the distributions
    """
    distributions = {}
    for definition in distributions_definition:
        if definition.name == "uniform":
            distributions[definition.name] = ephys.parameterscalers.NrnSegmentLinearScaler()
        else:
            distributions[definition.name] = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                name=definition.name,
                distribution=definition.function,
                dist_param_names=definition.parameters,
                soma_ref_location=definition.soma_ref_location,
            )
    return distributions


def _multi_locations(section_name, additional_multiloc_map):
    """Define a list of locations from a section names."""
    if additional_multiloc_map is not None:
        multiloc_map.update(additional_multiloc_map)
    return [
        ephys.locations.NrnSeclistLocation(sec, seclist_name=sec)
        for sec in multiloc_map.get(section_name, [section_name])
    ]


def _define_parameters(parameters_definition, distributions, mapping_multilocation):
    """Define a list of NrnParameter from a definition dictionary

    Args:
        parameters_definition (list): definitions of the parameters
        distributions (list): list of distributions in the form of ParameterScaler
        mapping_multilocation (dict): mapping from multi-locations names to list of locations
    """
    parameters = []
    for param_def in parameters_definition:
        if isinstance(param_def.value, (list, tuple)):
            is_frozen = False
            value = None
            bounds = param_def.value
            if bounds[0] > bounds[1]:
                raise ValueError(
                    f"Lower bound ({bounds[0]}) is greater than upper bound ({bounds[1]})"
                    f" for parameter {param_def.name}"
                )
        else:
            is_frozen = True
            value = param_def.value
            bounds = None

        if param_def.location == "global":
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_def.name,
                    param_name=param_def.name,
                    frozen=is_frozen,
                    bounds=bounds,
                    value=value,
                )
            )
            continue

        dist = None
        seclist_locations = None
        if "distribution_" in param_def.location:
            dist = distributions[param_def.location.split("distribution_")[1]]
        else:
            seclist_locations = _multi_locations(param_def.location, mapping_multilocation)

        if dist:
            parameters.append(
                ephys.parameters.MetaParameter(
                    name=f"{param_def.name}.{param_def.location}",
                    obj=dist,
                    attr_name=param_def.name,
                    frozen=is_frozen,
                    bounds=bounds,
                    value=value,
                )
            )
        elif param_def.distribution != "uniform":
            parameters.append(
                ephys.parameters.NrnRangeParameter(
                    name=f"{param_def.name}.{param_def.location}",
                    param_name=param_def.name,
                    value_scaler=distributions[param_def.distribution],
                    value=value,
                    bounds=bounds,
                    frozen=is_frozen,
                    locations=seclist_locations,
                )
            )
        else:
            parameters.append(
                ephys.parameters.NrnSectionParameter(
                    name=f"{param_def.name}.{param_def.location}",
                    param_name=param_def.name,
                    value_scaler=distributions["uniform"],
                    value=value,
                    bounds=bounds,
                    frozen=is_frozen,
                    locations=seclist_locations,
                )
            )

    return parameters


def _define_mechanisms(mechanisms_definition, mapping_multilocation):
    """Define a list of NrnMODMechanism from a definition dictionary

    Args:
        mechanisms_definition (list of MechanismConfiguration): definition of the mechanisms
        mapping_multilocation (dict): mapping from multi-locations names to list of locations

    Returns:
        list: list of NrnMODMechanism
    """
    mechanisms = []
    for mech_def in mechanisms_definition:
        seclist_locations = _multi_locations(mech_def.location, mapping_multilocation)
        mechanisms.append(
            ephys.mechanisms.NrnMODMechanism(
                name=f"{mech_def.name}.{mech_def.location}",
                mod_path=None,
                prefix=mech_def.name,
                locations=seclist_locations,
                preloaded=True,
                deterministic=not mech_def.stochastic,
            )
        )
    return mechanisms


def _define_morphology(
    model_configuration,
    nseg_frequency=40,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
):
    """Define a morphology object from a morphology file

    Args:
        model_configuration (NeuronModelConfiguration): configuration of the model
        nseg_frequency (float): frequency of nseg
        morph_modifiers (list): list of functions to modify the icell
                with (sim, icell) as arguments,
                if None, evaluation.modifiers.replace_axon_with_taper will be used
        morph_modifiers_hoc (list): list of hoc strings corresponding
                to morph_modifiers, each modifier can be a function, or a list of a path
                to a .py and the name of the function to use in this file

    Returns:
        bluepyopt.ephys.morphologies.NrnFileMorphology: a morphology object
    """
    if not morph_modifiers or morph_modifiers is None:
        morph_modifiers = [modifiers.replace_axon_with_taper]
        logger.debug("No morphology modifiers provided, replace_axon_with_taper will be used.")
    else:
        if isinstance(morph_modifiers, str):
            morph_modifiers = [morph_modifiers]
        for i, morph_modifier in enumerate(morph_modifiers):
            if isinstance(morph_modifier, list):
                modifier_module = importlib.import_module(morph_modifier[0])
                morph_modifiers[i] = getattr(modifier_module, morph_modifier[1])
            elif isinstance(morph_modifier, str):
                morph_modifiers[i] = getattr(modifiers, morph_modifier)
            elif not callable(morph_modifier):
                raise TypeError(
                    "A morph modifier is not callable nor a string nor a list of two str"
                )
    return NrnFileMorphology(
        morphology_path=model_configuration.morphology.path,
        do_replace_axon=False,
        do_set_nseg=True,
        nseg_frequency=nseg_frequency,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )


def create_cell_model(
    name,
    model_configuration,
    morph_modifiers=None,
    morph_modifiers_hoc=None,
    seclist_names=None,
    secarray_names=None,
    nseg_frequency=40,
):
    """Create a cell model based on a morphology, mechanisms and parameters

    Args:
        name (str): name of the model
        morphology (dict): morphology from emodel api .get_morphologies()
        model_configuration (NeuronModelConfiguration): Configuration of the neuron model,
            containing the parameters their locations and the associated mechanisms.
        morph_modifiers (list): list of functions to modify morphologies
        morph_modifiers_hoc (list): list of hoc functions to modify morphologies

    Returns:
        CellModel
    """
    morph = _define_morphology(
        model_configuration,
        nseg_frequency=nseg_frequency,
        morph_modifiers=morph_modifiers,
        morph_modifiers_hoc=morph_modifiers_hoc,
    )

    if seclist_names is None:
        seclist_names = model_configuration.morphology.seclist_names
    if secarray_names is None:
        secarray_names = model_configuration.morphology.secarray_names

    mechanisms = _define_mechanisms(
        model_configuration.mechanisms, model_configuration.mapping_multilocation
    )
    distributions = _define_distributions(model_configuration.distributions)
    parameters = _define_parameters(
        model_configuration.parameters, distributions, model_configuration.mapping_multilocation
    )
    return ephys.models.CellModel(
        name=name.replace(":", "_").replace("-", "_"),
        morph=morph,
        mechs=mechanisms,
        params=parameters,
        seclist_names=seclist_names,
        secarray_names=secarray_names,
    )


def _define_location(definition):
    """define location."""
    if definition["type"] == "CompRecording":
        if definition["location"] == "soma":
            return soma_loc
        if definition["location"] == "ais":
            return ais_loc
        raise ValueError("Only soma and ais are implemented for CompRecording")

    if definition["type"] == "somadistance":
        return NrnSomaDistanceCompLocation(
            name=definition["name"],
            soma_distance=definition["somadistance"],
            seclist_name=definition["seclist_name"],
        )

    if definition["type"] == "somadistanceapic":
        return NrnTrunkSomaDistanceCompLocation(
            name=definition["name"],
            soma_distance=definition["somadistance"],
            seclist_name=definition["seclist_name"],
        )

    if definition["type"] == "nrnseclistcomp":
        return NrnSeclistCompLocation(
            name=definition["name"],
            comp_x=definition["comp_x"],
            sec_index=definition["sec_index"],
            seclist_name=definition["seclist_name"],
        )

    raise ValueError(f"Unknown recording type {definition['type']}")


def _define_recording(recording_conf, use_fixed_dt_recordings=False):
    """Create a recording from a configuration dictionary

    Args:
        recording_conf (dict): configuration of the recording. Must contain the type of the
            recording as well as information about the location of the recording (see function
             define_location).
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.

    Returns:
        FixedDtRecordingCustom or LooseDtRecordingCustom
    """
    location = _define_location(recording_conf)
    variable = recording_conf.get("variable", recording_conf.get("var"))

    if use_fixed_dt_recordings:
        rec_class = bpopt.FixedDtRecordingCustom
    else:
        rec_class = bpopt.LooseDtRecordingCustom
    return rec_class(name=recording_conf["name"], location=location, variable=variable)


def _define_RMP_protocol(efeatures, stimulus_duration=500.0):
    """Define the resting membrane potential protocol"""
    target_voltage = None
    for f in efeatures:
        if (
            "RMPProtocol" in f.recording_names[""]
            and f.efel_feature_name == "steady_state_voltage_stimend"
        ):
            target_voltage = f
            break

    if not target_voltage:
        raise ValueError(
            "Couldn't find the efeature 'steady_state_voltage_stimend' associated to the "
            "'RMPProtocol' in your FitnessCalculatorConfiguration. It might not have been "
            "extracted from the ephys data you have available or the name of the protocol to"
            " use for RMP (setting 'name_rmp_protocol') might be wrong."
        )

    rmp_protocol = bpopt.RMPProtocol(
        name="RMPProtocol",
        location=soma_loc,
        target_voltage=target_voltage,
        stimulus_duration=stimulus_duration,
    )

    for f in efeatures:
        if (
            "RMPProtocol" in f.recording_names[""]
            and f.efel_feature_name != "steady_state_voltage_stimend"
        ):
            f.stim_start = 0.0
            f.stim_end = rmp_protocol.stimulus_duration
            f.stimulus_current = 0.0
    return rmp_protocol


def _define_Rin_protocol(
    efeatures,
    ais_recording=False,
    amp=-0.02,
    stimulus_delay=500.0,
    stimulus_duration=500.0,
    totduration=1000.0,
):
    """Define the input resistance protocol"""

    target_rin = None
    for f in efeatures:
        if (
            "RinProtocol" in f.recording_names[""]
            and f.efel_feature_name == "ohmic_input_resistance_vb_ssse"
        ):
            target_rin = f
            break

    if not target_rin:
        raise ValueError(
            "Couldn't find the efeature 'ohmic_input_resistance_vb_ssse' associated to "
            "the 'RinProtocol' in your FitnessCalculatorConfiguration. It might not have"
            "been extracted from the ephys data you have available or the name of the"
            " protocol to use for Rin (setting 'name_Rin_protocol') might be wrong."
        )

    location = soma_loc if not ais_recording else ais_loc

    return bpopt.RinProtocol(
        name="RinProtocol",
        location=location,
        target_rin=target_rin,
        amp=amp,
        stimulus_delay=stimulus_delay,
        stimulus_duration=stimulus_duration,
        totduration=totduration,
    )


def _define_holding_protocol(
    efeatures, strict_bounds=False, ais_recording=False, max_depth=7, stimulus_duration=500.0
):
    """Define the search holding current protocol"""
    target_voltage = None
    for f in efeatures:
        if (
            "SearchHoldingCurrent" in f.recording_names[""]
            and f.efel_feature_name == "steady_state_voltage_stimend"
        ):
            target_voltage = f
            break

    if target_voltage:
        return bpopt.SearchHoldingCurrent(
            name="SearchHoldingCurrent",
            location=soma_loc if not ais_recording else ais_loc,
            target_voltage=target_voltage,
            strict_bounds=strict_bounds,
            max_depth=max_depth,
            stimulus_duration=stimulus_duration,
        )

    raise ValueError(
        "Couldn't find the efeature 'bpo_holding_current' associated to "
        "the 'SearchHoldingCurrent' protocol in your FitnessCalculatorConfiguration."
    )


def _define_threshold_protocol(
    efeatures,
    max_threshold_voltage=-30,
    step_delay=500.0,
    step_duration=2000.0,
    totduration=3000.0,
    spikecount_timeout=50,
    max_depth=10,
):
    """Define the search threshold current protocol"""

    target_current = []
    for f in efeatures:
        if (
            "SearchThresholdCurrent" in f.recording_names[""]
            and f.efel_feature_name == "bpo_threshold_current"
        ):
            target_current.append(f)

    if target_current:
        return bpopt.SearchThresholdCurrent(
            name="SearchThresholdCurrent",
            location=soma_loc,
            target_threshold=target_current[0],
            max_threshold_voltage=max_threshold_voltage,
            stimulus_delay=step_delay,
            stimulus_duration=step_duration,
            stimulus_totduration=totduration,
            spikecount_timeout=spikecount_timeout,
            max_depth=max_depth,
        )

    raise ValueError(
        "Couldn't find the efeature 'bpo_threshold_current' or "
        "'bpo_holding_current' associated to the 'SearchHoldingCurrent'"
        " in your FitnessCalculatorConfiguration. It might not have"
        "been extracted from the ephys data you have available or the name of the"
        " protocol to use for Rin (setting 'name_Rin_protocol') might be wrong."
    )


def _define_protocols(
    fitness_calculator_configuration,
    include_validation_protocols,
    stochasticity,
    use_fixed_dt_recordings,
):
    """Instantiate several efeatures"""
    protocols = {}
    for protocols_def in fitness_calculator_configuration.protocols:
        if not include_validation_protocols and protocols_def.validation:
            continue
        protocols[protocols_def.name] = _define_protocol(
            protocols_def, stochasticity, use_fixed_dt_recordings
        )

    return protocols


class ProtocolConfiguration:

    """Container for the definition of a protocol"""

    def __init__(
        self,
        name,
        stimuli,
        recordings_from_config=None,
        recordings=None,
        validation=False,
        ion_variables=None,
        protocol_type="ThresholdBasedProtocol",
        stochasticity=False,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            name (str): name of the protocol
            stimuli (list of dict): contains the description of the stimuli. The exact format has
                to match what is expected by the related eCode class (see the classes defined
                in bluepyemodel.ecodes for more information). For example, for a Step protocol,
                the format will be:
                [{
                    'amp': float, 'thresh_perc': float, 'holding_current': float, 'delay': float,
                    'duration': float, 'totduration': float
                }]
            recordings_from_config (list of dict): contains the description of the recordings.
                For a recording at a given compartment, the format is for example:
                [{
                    "type": "CompRecording",
                    "name": f"{protocol_name}.soma.v",
                    "location": "soma",
                    "variable": "v",
                }]
            recordings (list of dict): same as recordings_from_config. Is here for backward
                compatibility only.
            ion_variables (list of str): ion current names and ionic concentration names
                for all available mechanisms.
            protocol_type (str): type of the protocol. Can be "ThresholdBasedProtocol" or
                "Protocol". When using "ThresholdBasedProtocol", the current amplitude and step
                amplitude of the stimulus will be ignored and replaced by values obtained from
                the holding current and rheobase of the cell model respectively. When using
                "Protocol", the current amplitude and step amplitude of the stimulus will be
                used directly, in this case, if a "thresh_perc" was informed, it will be ignored.
            stochasticity (bool): whether the mechanisms should be on stochastic mode
                when the protocol runs, or not.
        """

        self.name = name

        self.stimuli = stimuli
        if isinstance(self.stimuli, dict):
            self.stimuli = [self.stimuli]

        if recordings_from_config is None:
            if recordings is None:
                raise ValueError("Expected recordings_from_config to be not None")
            recordings_from_config = recordings
        if isinstance(recordings_from_config, dict):
            recordings_from_config = [recordings_from_config]

        self.recordings = []
        self.recordings_from_config = []
        for recording in recordings_from_config:
            self.recordings.append(recording)
            self.recordings_from_config.append(recording)

            if ion_variables is not None:
                for ion in ion_variables:
                    new_rec = recording.copy()

                    # it seems to only work without mech name at the end
                    if ion.startswith("i"):
                        ion = ion.split("_")[0]

                    if "variable" in recording:
                        new_rec["variable"] = ion
                    elif "var" in recording:
                        new_rec["var"] = ion
                    else:
                        raise KeyError("Expected 'var' or 'variable' in recording list.")

                    new_rec["name"] = ".".join(new_rec["name"].split(".")[:-1] + [ion])

                    self.recordings.append(new_rec)

        self.validation = validation
        self.protocol_type = protocol_type

        self.stochasticity = stochasticity

    def as_dict(self):
        """Dictionary form"""

        prot_as_dict = deepcopy(vars(self))
        prot_as_dict.pop("recordings")
        return prot_as_dict


def _define_protocol(protocol_configuration, stochasticity=False, use_fixed_dt_recordings=False):
    """Create a protocol.

    Args:
        protocol_configuration (ProtocolConfiguration): configuration of the protocol
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.

    Returns:
        Protocol
    """
    recordings = []
    for rec_conf in protocol_configuration.recordings:
        recordings.append(_define_recording(rec_conf, use_fixed_dt_recordings))

    if len(protocol_configuration.stimuli) != 1:
        raise ValueError("Only protocols with a single stimulus implemented")

    for k, ecode in eCodes.items():
        if k in protocol_configuration.name.lower():
            stimulus = ecode(location=soma_loc, **protocol_configuration.stimuli[0])
            break
    else:
        raise KeyError(
            f"There is no eCode linked to the stimulus name {protocol_configuration.name.lower()}. "
            "See ecode/__init__.py for the available stimuli "
            "names"
        )

    stoch = stochasticity and protocol_configuration.stochasticity

    if protocol_configuration.protocol_type in protocol_type_to_class:
        return protocol_type_to_class[protocol_configuration.protocol_type](
            name=protocol_configuration.name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=not stoch,
            stochasticity=stoch,
        )

    raise ValueError(f"Protocol type {protocol_configuration.protocol_type} not found")


def _format_protocol_name_to_list(protocol_name):
    """Make sure that the name of a protocol is a list [protocol_name, amplitude]"""

    if isinstance(protocol_name, str):
        try:
            name_parts = protocol_name.split("_")
            if name_parts[-1] == "hyp":
                amplitude = float(name_parts[-2])
                name_parts.pop(-2)
                name = "_".join(name_parts)
            else:
                name = "_".join(name_parts[:-1])
                amplitude = float(protocol_name.split("_")[-1])
        except ValueError:
            return protocol_name, None
        return name, amplitude

    if isinstance(protocol_name, list):
        return protocol_name

    raise TypeError("protocol_name should be a string or a list.")


def _are_same_protocol(name_a, name_b):
    """Check if two protocol names or list are equal. Eg: is IV_0.0 the same as IV_0 and
    the same as ["IV", 0.0]."""

    if name_a is None or name_b is None:
        return False

    amps = []
    ecodes = []

    for name in [name_a, name_b]:
        tmp_p = _format_protocol_name_to_list(name)
        ecodes.append(tmp_p[0])
        amps.append(tmp_p[1])

    if ecodes[0] == ecodes[1] and np.isclose(amps[0], amps[1]):
        return True
    return False


def _define_efeatures(
    fitness_calculator_configuration, include_validation_protocols, protocols, efel_settings
):
    """Instantiate several Protocols"""

    efeatures = []
    validation_prot = fitness_calculator_configuration.validation_protocols

    for feature_def in fitness_calculator_configuration.efeatures:
        if not include_validation_protocols and any(
            _are_same_protocol(feature_def.protocol_name, p) for p in validation_prot
        ):
            continue

        protocol = None
        if feature_def.protocol_name not in PRE_PROTOCOLS:
            protocol = next(
                (p for p in protocols.values() if p.name == feature_def.protocol_name), None
            )
            if protocol is not None:
                # here it is for validation protocols, to clean up
                efeatures.append(_define_efeature(feature_def, protocol, efel_settings))
        else:
            efeatures.append(_define_efeature(feature_def, protocol, efel_settings))

    return efeatures


def _define_efeature(feature_config, protocol=None, global_efel_settings=None):
    """Define an efeature from a configuration dictionary"""

    global_efel_settings = {} if global_efel_settings is None else global_efel_settings

    stim_amp = None
    stim_start = None
    stim_end = None

    if protocol:
        stim_amp = protocol.amplitude

    if feature_config.efel_settings.get("stim_start", None) is not None:
        stim_start = feature_config.efel_settings["stim_start"]
    elif protocol:
        stim_start = protocol.stim_start()

    if feature_config.efel_settings.get("stim_end", None) is not None:
        stim_end = feature_config.efel_settings["stim_end"]
    elif protocol:
        if "bAP" in protocol.name:
            stim_end = protocol.total_duration
        else:
            stim_end = protocol.stim_end()

    efel_settings = {**global_efel_settings, **feature_config.efel_settings}

    # Handle the special case of multiple_decay_time_constant_after_stim
    if feature_config.efel_feature_name == "multiple_decay_time_constant_after_stim":
        if hasattr(protocol.stimuli[0], "multi_stim_start"):
            efel_settings["multi_stim_start"] = protocol.stimuli[0].multi_stim_start()
            efel_settings["multi_stim_end"] = protocol.stimuli[0].multi_stim_end()
        else:
            efel_settings["multi_stim_start"] = [stim_start]
            efel_settings["multi_stim_end"] = [stim_end]
    double_settings = {k: v for k, v in efel_settings.items() if isinstance(v, (float, list))}
    int_settings = {k: v for k, v in efel_settings.items() if isinstance(v, int)}
    string_settings = {k: v for k, v in efel_settings.items() if isinstance(v, str)}

    efeature = bpopt.eFELFeatureBPEM(
        feature_config.name,
        efel_feature_name=feature_config.efel_feature_name,
        recording_names=feature_config.recording_name_for_instantiation,
        stim_start=stim_start,
        stim_end=stim_end,
        exp_mean=feature_config.mean,
        exp_std=feature_config.std,
        stimulus_current=stim_amp,
        threshold=efel_settings.get("Threshold", None),
        interp_step=efel_settings.get("interp_step", None),
        double_settings=double_settings,
        int_settings=int_settings,
        string_settings=string_settings,
    )

    return efeature


def _set_morphology_dependent_locations(stimulus, cell):
    """Here we deal with morphology dependent locations"""

    def _get_stim(stimulus, sec_id):
        new_stim = deepcopy(stimulus)
        stim_split = stimulus["name"].split(".")
        new_stim["type"] = "nrnseclistcomp"
        new_stim["name"] = f"{'.'.join(stim_split[:-1])}_{sec_id}.{stim_split[-1]}"
        new_stim["sec_index"] = sec_id
        return new_stim

    new_stims = []
    if stimulus["type"] == "somadistanceapic":
        new_stims = [deepcopy(stimulus)]
        new_stims[0]["sec_name"] = seclist_to_sec.get(
            stimulus["seclist_name"], stimulus["seclist_name"]
        )

    elif stimulus["type"] == "terminal_sections":
        # all terminal sections
        for sec_id, section in enumerate(getattr(cell.icell, stimulus["seclist_name"])):
            if len(section.subtree()) == 1:
                new_stims.append(_get_stim(stimulus, sec_id))

    elif stimulus["type"] == "all_sections":
        # all section of given type
        for sec_id, section in enumerate(getattr(cell.icell, stimulus["seclist_name"])):
            new_stims.append(_get_stim(stimulus, sec_id))

    else:
        new_stims = [deepcopy(stimulus)]

    if len(new_stims) == 0 and stimulus["type"] in [
        "somadistanceapic",
        "terminal_sections",
        "all_sections",
    ]:
        logger.warning("We could not add a location for %s", stimulus)
    return new_stims


class EFeatureConfiguration:

    """Container for the definition of an EFeature"""

    def __init__(
        self,
        efel_feature_name,
        protocol_name,
        recording_name,
        mean,
        efeature_name=None,
        efel_settings=None,
        threshold_efeature_std=None,
        original_std=None,
        std=None,
        sample_size=None,
        default_std_value=1e-3,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efel_feature_name (str): name of the eFEl feature.
            protocol_name (str): name of the protocol to which the efeature is associated. For
                example "Step_200".
            recording_name (str or dict): name of the recording(s) of the protocol. For
                example: "soma.v" or if and only if the feature depends on several recordings:
                {"": "soma.v", "location_AIS": "axon.v"}.
            mean (float): mean of the efeature.
            original_std (float): unmodified standard deviation of the efeature
            std (float): kept for legacy purposes.
            efeature_name (str):given name for this specific feature. Can be different
                from the efel efeature name.
            efel_settings (dict): eFEl settings.
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional).
            sample_size (float): number of data point that were used to compute the present
                average and standard deviation.
        """

        self.efel_feature_name = efel_feature_name
        self.protocol_name = protocol_name
        self.recording_name = recording_name
        self.threshold_efeature_std = threshold_efeature_std
        self.default_std_value = default_std_value

        self.mean = mean
        self.original_std = original_std if original_std is not None else std
        self.sample_size = sample_size

        self.efeature_name = efeature_name

        if efel_settings is None:
            self.efel_settings = {"strict_stiminterval": True}
        else:
            self.efel_settings = efel_settings

    @property
    def name(self):
        """ """
        n = self.efeature_name if self.efeature_name else self.efel_feature_name
        if isinstance(self.recording_name, dict):
            return f"{self.protocol_name}.{self.recording_name['']}.{n}"
        return f"{self.protocol_name}.{self.recording_name}.{n}"

    @property
    def recording_name_for_instantiation(self):
        """ """
        if isinstance(self.recording_name, dict):
            return {k: f"{self.protocol_name}.{v}" for k, v in self.recording_name.items()}
        return {"": f"{self.protocol_name}.{self.recording_name}"}

    @property
    def std(self):
        """Limit the standard deviation with a lower bound equal to a percentage of the mean."""

        if self.threshold_efeature_std is None:
            return self.original_std

        if self.mean == 0.0:
            if self.threshold_efeature_std:
                return self.threshold_efeature_std
            return self.default_std_value

        limit = abs(self.threshold_efeature_std * self.mean)
        if self.original_std < limit:
            return limit

        return self.original_std

    def as_dict(self):
        """Dictionary form"""

        return vars(self)


class FitnessCalculatorConfiguration:

    """The goal of this class is to store the results of an efeature extraction (efeatures
    and protocols) or to contain the results of a previous extraction retrieved from an access
    point. This object is used for the creation of the fitness calculator.
    """

    def __init__(
        self,
        efeatures=None,
        protocols=None,
        name_rmp_protocol=None,
        name_rin_protocol=None,
        threshold_efeature_std=None,
        default_std_value=1e-3,
        validation_protocols=None,
        stochasticity=False,
        ion_variables=None,
    ):
        """Init.

        The arguments efeatures and protocols are expected to be in the format used for the
        storage of the fitness calculator configuration. To store the results of an extraction,
        use the method init_from_bluepyefe.

        Args:
            efeatures (list of dict): contains the description of the efeatures of the model
                in the format returned by the method as_dict of the EFeatureConfiguration class:
                [
                    {"efel_feature_name": str, "protocol_name": str, "recording_name": str,
                    "mean": float, "std": float, "efel_settings": dict}
                ]
            protocols (list of dict): contains the description of the protocols of the model
                in the format returned by the method as_dict of the ProtocolConfiguration class:
                [
                    {"name": str, "stimuli": list of dict, "recordings": list of dict,
                    "validation": bool}
                ]
            name_rmp_protocol (str or list): name and amplitude of protocol
                whose features are to be used as targets for the search of the RMP.
                e.g: ["IV", 0] or "IV_0"
            name_rin_protocol (str or list): name and amplitude of protocol
                whose features are to be used as targets for the search of the Rin.
                e.g: ["IV", -20] or "IV_-20"
            threshold_efeature_std (float): lower limit for the std expressed as a percentage of
                the mean of the features value (optional). Legacy.
             default_std_value (float): during and after extraction, this value will be used
                to replace the standard deviation if the standard deviation is 0.
            validation_protocols (list of str): name of the protocols used for validation only.
            stochasticity (bool or list of str): should channels behave stochastically if they can.
                If a list of protocol names is provided, the runs will be stochastic
                for these protocols, and deterministic for the other ones.
            ion_variables (list of str): ion current names and ionic concentration names
                for all available mechanisms
        """
        self.rmp_duration = 500.0
        self.rin_step_delay = 500.0
        self.rin_step_duration = 500.0
        self.rin_step_amp = -0.02
        self.rin_totduration = self.rin_step_delay + self.rin_step_duration
        self.search_holding_duration = 500.0
        self.search_threshold_step_delay = 500.0
        self.search_threshold_step_duration = 2000.0
        self.search_threshold_totduration = 3000.0
        self.ion_variables = ion_variables

        if protocols is None:
            self.protocols = []
        else:
            self.protocols = [
                ProtocolConfiguration(**p, ion_variables=self.ion_variables) for p in protocols
            ]

        self.efeatures = []
        if efeatures is not None:
            for f in efeatures:
                f_dict = deepcopy(f)
                f_dict.pop("threshold_efeature_std", None)
                f_dict.pop("default_std_value", None)
                self.efeatures.append(
                    EFeatureConfiguration(
                        **f_dict,
                        threshold_efeature_std=f.get(
                            "threshold_efeature_std", threshold_efeature_std
                        ),
                        default_std_value=f.get("default_std_value", default_std_value),
                    )
                )

        if validation_protocols is None:
            self.validation_protocols = []
        else:
            self.validation_protocols = validation_protocols

        self.stochasticity = stochasticity

        self.name_rmp_protocol = name_rmp_protocol
        self.name_rin_protocol = name_rin_protocol

    def protocol_exist(self, protocol_name):
        """ """
        return bool(p for p in self.protocols if p.name == protocol_name)

    def check_stochasticity(self, protocol_name):
        """Check if stochasticity should be active for a given protocol"""
        if isinstance(self.stochasticity, list):
            return protocol_name in self.stochasticity
        return self.stochasticity

    def _add_bluepyefe_protocol(self, protocol_name, protocol):
        """"""
        # By default include somatic recording
        recordings = [
            {
                "type": "CompRecording",
                "name": f"{protocol_name}.soma.v",
                "location": "soma",
                "variable": "v",
            }
        ]

        stimulus = deepcopy(protocol["step"])
        stimulus["holding_current"] = protocol["holding"]["amp"]

        validation = any(_are_same_protocol(protocol_name, p) for p in self.validation_protocols)
        stochasticity = self.check_stochasticity(protocol_name)

        protocol_type = "Protocol"
        if self.name_rmp_protocol and self.name_rin_protocol:
            protocol_type = "ThresholdBasedProtocol"

        tmp_protocol = ProtocolConfiguration(
            name=protocol_name,
            stimuli=[stimulus],
            recordings_from_config=recordings,
            validation=validation,
            ion_variables=self.ion_variables,
            stochasticity=stochasticity,
            protocol_type=protocol_type,
        )

        self.protocols.append(tmp_protocol)

    def _add_bluepyefe_efeature(self, feature, protocol_name, recording, threshold_efeature_std):
        """"""

        recording_name = "soma.v" if recording == "soma" else recording

        tmp_feature = EFeatureConfiguration(
            efel_feature_name=feature["feature"],
            protocol_name=protocol_name,
            recording_name=recording_name,
            mean=feature["val"][0],
            std=feature["val"][1],
            efeature_name=feature.get("efeature_name", None),
            efel_settings=feature.get("efel_settings", {}),
            threshold_efeature_std=threshold_efeature_std,
            sample_size=feature.get("n", None),
        )

        if (
            _are_same_protocol(self.name_rmp_protocol, protocol_name)
            and feature["feature"] == "voltage_base"
        ):
            tmp_feature.protocol_name = "RMPProtocol"
            tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
        if (
            _are_same_protocol(self.name_rin_protocol, protocol_name)
            and feature["feature"] == "voltage_base"
        ):
            tmp_feature.protocol_name = "SearchHoldingCurrent"
            tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
        if (
            _are_same_protocol(self.name_rin_protocol, protocol_name)
            and feature["feature"] == "ohmic_input_resistance_vb_ssse"
        ):
            tmp_feature.protocol_name = "RinProtocol"

        if protocol_name not in PRE_PROTOCOLS and not self.protocol_exist(protocol_name):
            raise ValueError(
                f"Trying to register efeatures for protocol {protocol_name},"
                " but this protocol does not exist"
            )

        self.efeatures.append(tmp_feature)

    def init_from_bluepyefe(self, efeatures, protocols, currents, threshold_efeature_std):
        """Fill the configuration using the output of BluePyEfe"""

        if self.name_rmp_protocol and not any(
            _are_same_protocol(self.name_rmp_protocol, p) for p in efeatures
        ):
            raise ValueError(
                f"The stimulus {self.name_rmp_protocol} requested for RMP "
                "computation couldn't be extracted from the ephys data."
            )
        if self.name_rin_protocol and not any(
            _are_same_protocol(self.name_rin_protocol, p) for p in efeatures
        ):
            raise ValueError(
                f"The stimulus {self.name_rin_protocol} requested for Rin "
                "computation couldn't be extracted from the ephys data."
            )

        self.protocols = []
        self.efeatures = []

        for protocol_name, protocol in protocols.items():
            self._add_bluepyefe_protocol(protocol_name, protocol)

        for protocol_name in efeatures:
            for recording in efeatures[protocol_name]:
                for feature in efeatures[protocol_name][recording]:
                    self._add_bluepyefe_efeature(
                        feature, protocol_name, recording, threshold_efeature_std
                    )

        # Add the current related features
        if currents and self.name_rmp_protocol and self.name_rin_protocol:
            self.efeatures.append(
                EFeatureConfiguration(
                    efel_feature_name="bpo_holding_current",
                    protocol_name="SearchHoldingCurrent",
                    recording_name="soma.v",
                    mean=currents["holding_current"][0],
                    std=currents["holding_current"][1],
                    threshold_efeature_std=threshold_efeature_std,
                )
            )

            self.efeatures.append(
                EFeatureConfiguration(
                    efel_feature_name="bpo_threshold_current",
                    protocol_name="SearchThresholdCurrent",
                    recording_name="soma.v",
                    mean=currents["threshold_current"][0],
                    std=currents["threshold_current"][1],
                    threshold_efeature_std=threshold_efeature_std,
                )
            )

        self.remove_featureless_protocols()

    def _add_legacy_protocol(self, protocol_name, protocol):
        """"""

        # By default include somatic recording
        recordings = [
            {
                "type": "CompRecording",
                "name": f"{protocol_name}.soma.v",
                "location": "soma",
                "variable": "v",
            }
        ]

        if "extra_recordings" in protocol:
            for protocol_def in protocol["extra_recordings"]:
                recordings.append(protocol_def)
                protocol_def[
                    "name"
                ] = f"{protocol_name}.{protocol_def['name']}.{protocol_def['var']}"

        stimulus = deepcopy(protocol["stimuli"]["step"])
        if "holding" in protocol["stimuli"]:
            stimulus["holding_current"] = protocol["stimuli"]["holding"]["amp"]
        else:
            stimulus["holding_current"] = None

        validation = any(_are_same_protocol(protocol_name, p) for p in self.validation_protocols)
        stochasticity = self.check_stochasticity(protocol_name)

        protocol_type = "Protocol"
        if "type" in protocol and protocol["type"] == "StepThresholdProtocol":
            protocol_type = "ThresholdBasedProtocol"

        tmp_protocol = ProtocolConfiguration(
            name=protocol_name,
            stimuli=[stimulus],
            recordings_from_config=recordings,
            validation=validation,
            ion_variables=self.ion_variables,
            protocol_type=protocol_type,
            stochasticity=stochasticity,
        )

        self.protocols.append(tmp_protocol)

    def _add_legacy_efeature(self, feature, protocol_name, recording, threshold_efeature_std):
        """"""

        recording_name = "soma.v" if recording == "soma" else recording

        tmp_feature = EFeatureConfiguration(
            efel_feature_name=feature["feature"],
            protocol_name=protocol_name,
            recording_name=recording_name,
            mean=feature["val"][0],
            std=feature["val"][1],
            efeature_name=feature.get("efeature_name", None),
            efel_settings=feature.get("efel_settings", {}),
            threshold_efeature_std=threshold_efeature_std,
        )

        if protocol_name == "Rin":
            if feature["feature"] == "ohmic_input_resistance_vb_ssse":
                tmp_feature.protocol_name = "RinProtocol"
            elif feature["feature"] == "voltage_base":
                tmp_feature.protocol_name = "SearchHoldingCurrent"
                tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
            else:
                return

        if protocol_name == "RMP":
            if feature["feature"] == "voltage_base":
                tmp_feature.protocol_name = "RMPProtocol"
                tmp_feature.efel_feature_name = "steady_state_voltage_stimend"
            elif feature["feature"] == "Spikecount":
                tmp_feature.protocol_name = "RMPProtocol"
                tmp_feature.efel_feature_name = "Spikecount"
            else:
                return

        if protocol_name == "RinHoldCurrent":
            tmp_feature.protocol_name = "SearchHoldingCurrent"
        if protocol_name == "Threshold":
            tmp_feature.protocol_name = "SearchThresholdCurrent"

        if protocol_name not in PRE_PROTOCOLS and not self.protocol_exist(protocol_name):
            raise ValueError(
                f"Trying to register efeatures for protocol {protocol_name},"
                " but this protocol does not exist"
            )

        self.efeatures.append(tmp_feature)

    def init_from_legacy_dict(self, efeatures, protocols, threshold_efeature_std):
        """ """
        self.protocols = []
        self.efeatures = []

        if (
            self.name_rmp_protocol
            and not any(_are_same_protocol(self.name_rmp_protocol, p) for p in efeatures)
            and "RMP" not in efeatures
        ):
            raise ValueError(
                f"The protocol {self.name_rmp_protocol} requested for RMP nor RMPProtocol "
                "are present in your efeatures json file."
            )

        if (
            self.name_rin_protocol
            and not any(_are_same_protocol(self.name_rin_protocol, p) for p in efeatures)
            and "Rin" not in efeatures
        ):
            raise ValueError(
                f"The protocol {self.name_rin_protocol} requested for Rin nor RinProtocol "
                "are present in your efeatures json file."
            )

        for protocol_name, protocol in protocols.items():
            if protocol_name == "RMP":
                self.rmp_duration = protocol["stimuli"]["step"]["duration"]
            if protocol_name == "Rin":
                self.rin_step_delay = protocol["stimuli"]["step"]["delay"]
                self.rin_step_duration = protocol["stimuli"]["step"]["duration"]
                self.rin_step_amp = protocol["stimuli"]["step"]["amp"]
                self.rin_totduration = protocol["stimuli"]["step"]["totduration"]
            if protocol_name == "ThresholdDetection":
                self.search_threshold_step_delay = protocol["step_template"]["stimuli"]["step"][
                    "delay"
                ]
                self.search_threshold_step_duration = protocol["step_template"]["stimuli"]["step"][
                    "duration"
                ]
                self.search_threshold_totduration = protocol["step_template"]["stimuli"]["step"][
                    "totduration"
                ]

            if protocol_name in PRE_PROTOCOLS + LEGACY_PRE_PROTOCOLS:
                continue

            self._add_legacy_protocol(protocol_name, protocol)
            validation = protocol.get("validation", False)

            if validation != self.protocols[-1].validation:
                raise ValueError(
                    "The protocol was set as a validation protocol in the json but is not present "
                    "as a validation protocol in the settings"
                )

        for protocol_name in efeatures:
            for recording in efeatures[protocol_name]:
                for feature in efeatures[protocol_name][recording]:
                    self._add_legacy_efeature(
                        feature, protocol_name, recording, threshold_efeature_std
                    )

        self.remove_featureless_protocols()

    def remove_featureless_protocols(self):
        """Remove the protocols that o not have any matching efeatures"""

        to_remove = []

        for i, protocol in enumerate(self.protocols):
            for efeature in self.efeatures:
                if efeature.protocol_name == protocol.name:
                    break
            else:
                to_remove.append(i)

        self.protocols = [p for i, p in enumerate(self.protocols) if i not in to_remove]

    def configure_morphology_dependent_locations(self, _cell, simulator):
        """"""

        cell = deepcopy(_cell)
        cell.params = None
        cell.mechanisms = None
        cell.instantiate(sim=simulator)

        # TODO: THE SAME FOR STIMULI

        for i, protocol in enumerate(self.protocols):
            recordings = []
            for j, rec in enumerate(protocol.recordings):
                if rec["type"] != "CompRecording":
                    for _rec in _set_morphology_dependent_locations(rec, cell):
                        recordings.append(_rec)
                else:
                    recordings.append(self.protocols[i].recordings[j])
            self.protocols[i].recordings = recordings

        # if the loc of the recording is of the form axon*.v, we replace * by
        # all the corresponding int from the created recordings
        to_remove = []
        efeatures = []
        for i, efeature in enumerate(self.efeatures):
            if isinstance(efeature.recording_name, str):
                loc_name, rec_name = efeature.recording_name.split(".")
                if loc_name[-1] == "*":
                    to_remove.append(i)
                    protocol = next(p for p in self.protocols if p.name == efeature.protocol_name)
                    for rec in protocol.recordings:
                        base_rec_name = rec["name"].split(".")[1]
                        if base_rec_name.startswith(loc_name[:-1]):
                            efeatures.append(deepcopy(efeature))
                            efeatures[-1].recording_name = f"{base_rec_name}.{rec_name}"

        self.efeatures = [f for i, f in enumerate(self.efeatures) if i not in to_remove] + efeatures

    def as_dict(self):
        """Used for the storage of the configuration"""

        return {
            "efeatures": [e.as_dict() for e in self.efeatures],
            "protocols": [p.as_dict() for p in self.protocols],
        }

    def __str__(self):
        """String representation"""

        str_form = "Fitness Calculator Configuration:\n\n"

        str_form += "Protocols:\n"
        for p in self.protocols:
            str_form += f"   {p.as_dict()}\n"

        str_form += "EFeatures:\n"
        for f in self.efeatures:
            str_form += f"   {f.as_dict()}\n"

        return str_form


def _define_main_protocol(
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    efel_settings=None,
    use_fixed_dt_recordings=False,
):
    """Create a meta protocol in charge of running the other protocols.

    Args:
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        efel_settings (dict): eFEl settings.
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.
    """
    protocols = _define_protocols(
        fitness_calculator_configuration,
        include_validation_protocols,
        stochasticity,
        use_fixed_dt_recordings,
    )

    efeatures = _define_efeatures(
        fitness_calculator_configuration, include_validation_protocols, protocols, efel_settings
    )
    return bpopt.ProtocolRunner(protocols), efeatures


def _define_threshold_based_main_protocol(
    fitness_calculator_configuration,
    include_validation_protocols=False,
    stochasticity=True,
    ais_recording=False,
    efel_settings=None,
    max_threshold_voltage=-30,
    strict_holding_bounds=True,
    use_fixed_dt_recordings=False,
    max_depth_holding_search=7,
    max_depth_threshold_search=10,
    spikecount_timeout=50,
):
    """Create a meta protocol in charge of running the other protocols.

    The amplitude of the "threshold_protocols" depend on the computation of
    the current threshold.

    Args:
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        stochasticity (bool): Should the stochastic channels be stochastic or
            deterministic
        ais_recording (bool): if True all the soma recording will be at the first axonal section.
        efel_settings (dict): eFEl settings.
        max_threshold_voltage (float): maximum voltage at which the SearchThresholdProtocol
            will search for the rheobase.
        strict_holding_bounds (bool): to adaptively enlarge bounds if holding current is outside
            when set to False
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.
        max_depth_holding_search (int): maximum depth for the binary search for the
            holding current
        max_depth_threshold_search (int): maximum depth for the binary search for the
            threshold current
        spikecount_timeout (float): timeout for spikecount computation, if timeout is reached,
            we set spikecount=2 as if many spikes were present, to speed up bisection search.
    """
    protocols = _define_protocols(
        fitness_calculator_configuration,
        include_validation_protocols,
        stochasticity,
        use_fixed_dt_recordings,
    )

    efeatures = _define_efeatures(
        fitness_calculator_configuration, include_validation_protocols, protocols, efel_settings
    )

    # Create the special protocols
    protocols.update(
        {
            "RMPProtocol": _define_RMP_protocol(
                efeatures, stimulus_duration=fitness_calculator_configuration.rmp_duration
            ),
            "SearchHoldingCurrent": _define_holding_protocol(
                efeatures,
                strict_holding_bounds,
                ais_recording,
                max_depth_holding_search,
                stimulus_duration=fitness_calculator_configuration.search_holding_duration,
            ),
            "RinProtocol": _define_Rin_protocol(
                efeatures,
                ais_recording,
                amp=fitness_calculator_configuration.rin_step_amp,
                stimulus_delay=fitness_calculator_configuration.rin_step_delay,
                stimulus_duration=fitness_calculator_configuration.rin_step_duration,
                totduration=fitness_calculator_configuration.rin_totduration,
            ),
            "SearchThresholdCurrent": _define_threshold_protocol(
                efeatures,
                max_threshold_voltage,
                fitness_calculator_configuration.search_threshold_step_delay,
                fitness_calculator_configuration.search_threshold_step_duration,
                fitness_calculator_configuration.search_threshold_totduration,
                spikecount_timeout,
                max_depth_threshold_search,
            ),
        }
    )

    return bpopt.ProtocolRunner(protocols), efeatures


def define_fitness_calculator(features):
    """Creates the objectives calculator.

    Args:
        features (list): list of EFeature.

    Returns:
        ObjectivesCalculator
    """

    objectives = [SingletonObjective(feat.name, feat) for feat in features]

    return ObjectivesCalculator(objectives)


def get_simulator(stochasticity, cell_model, dt=None, cvode_minstep=0.0):
    """Get NrnSimulator

    Args:
        stochasticity (bool): allow the use of simulator for stochastic channels
        cell_model (CellModel): used to check if any stochastic channels are present
        dt (float): if not None, cvode will be disabled and fixed timesteps used.
        cvode_minstep (float): minimum time step allowed for a CVODE step.
    """
    # set smaller tolerance to handle michaelis-mentens term with cvode
    if os.environ.get("MM_CVODE"):
        import neuron  # pylint: disable=import-outside-toplevel

        cvode = neuron.h.CVode()
        cvode.atolscale("cai", 1e-8)
        cvode.atol(1e-8)

    if stochasticity:
        for mechanism in cell_model.mechanisms:
            if not mechanism.deterministic:
                return NrnSimulator(dt=dt or 0.025, cvode_active=False)

        logger.warning(
            "Stochasticity is True but no mechanisms are stochastic. Switching to "
            "non-stochastic."
        )

    if dt is None:
        return NrnSimulator(cvode_minstep=cvode_minstep)

    return NrnSimulator(dt=dt, cvode_active=False)


def create_evaluator(
    cell_model,
    fitness_calculator_configuration,
    settings,
    stochasticity=None,
    timeout=None,
    include_validation_protocols=False,
    use_fixed_dt_recordings=False,
):
    """Creates an evaluator for a cell model/protocols/e-feature combo.

    Args:
        cell_model (CellModel): cell model
        fitness_calculator_configuration (FitnessCalculatorConfiguration): configuration of the
            fitness calculator.
        settings (dict): settings for the pipeline.
        stochasticity (bool): should the stochastic channels be stochastic or
            deterministic
        timeout (float): maximum time in second during which a protocol is
            allowed to run
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.

    Returns:
        CellEvaluator
    """

    stochasticity = settings.get("stochasticity", False) if stochasticity is None else stochasticity

    simulator = get_simulator(
        stochasticity=stochasticity,
        cell_model=cell_model,
        dt=settings.get("neuron_dt", None),
        cvode_minstep=settings.get("cvode_minstep", 0.0),
    )

    fitness_calculator_configuration.configure_morphology_dependent_locations(cell_model, simulator)

    if any(
        p.protocol_type == "ThresholdBasedProtocol"
        for p in fitness_calculator_configuration.protocols
    ):
        main_protocol, features = _define_threshold_based_main_protocol(
            fitness_calculator_configuration,
            include_validation_protocols,
            stochasticity=stochasticity,
            efel_settings=settings.get(
                "efel_settings", {"interp_step": 0.025, "strict_stiminterval": True}
            ),
            max_threshold_voltage=settings.get("max_threshold_voltage", -30),
            strict_holding_bounds=settings.get("strict_holding_bounds", True),
            use_fixed_dt_recordings=use_fixed_dt_recordings,
            max_depth_holding_search=settings.get("max_depth_holding_search", 7),
            max_depth_threshold_search=settings.get("max_depth_threshold_search", 10),
            spikecount_timeout=settings.get("spikecount_timeout", 50),
        )
    else:
        main_protocol, features = _define_main_protocol(
            fitness_calculator_configuration,
            include_validation_protocols,
            stochasticity=stochasticity,
            efel_settings=settings.get(
                "efel_settings", {"interp_step": 0.025, "strict_stiminterval": True}
            ),
            use_fixed_dt_recordings=use_fixed_dt_recordings,
        )

    fitness_calculator = define_fitness_calculator(features)
    fitness_protocols = {"main_protocol": main_protocol}

    param_names = [param.name for param in cell_model.params.values() if not param.frozen]

    cell_eval = CellEvaluator(
        cell_model=cell_model,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=simulator,
        use_params_for_seed=True,
        timeout=settings.get("optimisation_timeout") if timeout is None else timeout,
    )
    cell_eval.prefix = cell_model.name

    return cell_eval


def get_evaluator_from_access_point(
    emodel,
    access_point,
    stochasticity=None,
    include_validation_protocols=False,
    timeout=None,
    use_fixed_dt_recordings=False,
    record_ions_and_currents=False,
):
    """Create an evaluator for the emodel.

    Args:
        access_point (DataAccessPoint): API used to access the database
        stochasticity (bool): should channels behave stochastically if they can.
        include_validation_protocols (bool): should the validation protocols
            and validation efeatures be added to the evaluator.
        timeout (float): duration (in second) after which the evaluation of a
            protocol will be interrupted.
        use_fixed_dt_recordings (bool): whether to record at a fixed dt of 0.1 ms.
        record_ions_and_currents (bool): whether to add the ion and non-specific currents
            and the ionic concentration to the recordings.

    Returns:
        bluepyopt.ephys.evaluators.CellEvaluator
    """
    settings = access_point.get_settings(emodel)
    configuration = access_point.get_configuration(emodel)
    calculator_configuration = access_point.get_calculator_configuration(
        emodel, record_ions_and_currents=record_ions_and_currents
    )
    cell_model = create_cell_model(
        name=emodel,
        model_configuration=configuration,
        morph_modifiers=settings.get("morph_modifiers", None),
    )

    return create_evaluator(
        cell_model=cell_model,
        fitness_calculator_configuration=calculator_configuration,
        settings=settings,
        stochasticity=stochasticity,
        timeout=timeout,
        include_validation_protocols=include_validation_protocols,
        use_fixed_dt_recordings=use_fixed_dt_recordings,
    )


def _single_feature_evaluation(
    combo,
    access_point=None,
    morphology_path="path",
    stochasticity=False,
    trace_data_path=None,
    threshold_only=False,
    record_ions_and_currents=False,
):
    """Evaluating single protocol and save traces."""
    if morphology_path in combo:
        access_point.morph_path = combo[morphology_path]

    access_point.settings = access_point.get_settings(combo["emodel"])

    if "morph_modifiers" not in access_point.settings:
        access_point.settings["morph_modifiers"] = None

    if "ais_model" in combo and isinstance(combo["ais_model"], str):
        if access_point.settings["morph_modifiers"] is None:
            access_point.settings["morph_modifiers"] = []
        access_point.settings["morph_modifiers"].append(
            partial(
                modifiers.synth_axon,
                params=json.loads(combo["ais_model"])["popt"],
                scale=combo["ais_scaler"],
            )
        )

    if "soma_model" in combo and isinstance(combo["soma_model"], str):
        if access_point.settings["morph_modifiers"] is None:
            access_point.settings["morph_modifiers"] = []
        access_point.settings["morph_modifiers"].append(
            partial(
                modifiers.synth_soma,
                params=json.loads(combo["soma_model"]),
                scale=combo["soma_scaler"],
            ),
        )

    if "axon_hillock" in combo and isinstance(combo["axon_hillock"], str):
        if access_point.settings["morph_modifiers"] is None:
            access_point.setting["morph_modifiers"] = []
        access_point.settings["morph_modifiers"] = [
            partial(modifiers.replace_axon_hillock, **json.loads(combo["axon_hillock"]))
        ]

    # we do that here to fetch parameters for emodel with seed
    access_point.settings["strict_holding_bounds"] = False
    access_point.settings["max_depth_holding_search"] = 20
    access_point.settings["max_depth_threshold_search"] = 30
    access_point.settings["spikecount_timeout"] = 2 * 60
    evaluator = get_evaluator_from_access_point(
        combo["emodel"],
        access_point,
        stochasticity=stochasticity,
        timeout=10000,
        record_ions_and_currents=record_ions_and_currents,
    )
    if "params" in access_point.final[combo["emodel"]]:
        params = access_point.final[combo["emodel"]]["params"]
    else:
        params = access_point.final[combo["emodel"]]["parameters"]

    if "new_parameters" in combo:
        params.update(json.loads(combo["new_parameters"]))

    if threshold_only:
        for prot in list(evaluator.fitness_protocols["main_protocol"].protocols):
            if prot not in [
                "SearchThresholdCurrent",
                "RinProtocol",
                "SearchHoldingCurrent",
                "RMPProtocol",
            ]:
                evaluator.fitness_protocols["main_protocol"].protocols.pop(prot)

                evaluator.fitness_protocols[
                    "main_protocol"
                ].execution_order = evaluator.fitness_protocols[
                    "main_protocol"
                ].compute_execution_order()

    evaluator.cell_model.unfreeze(params)
    responses = evaluator.run_protocols(evaluator.fitness_protocols.values(), params)
    features = evaluator.fitness_calculator.calculate_values(responses)
    scores = evaluator.fitness_calculator.calculate_scores(responses)

    for f, val in features.items():
        if isinstance(val, np.ndarray) and len(val) > 0:
            try:
                features[f] = np.nanmean(val)
            except (AttributeError, TypeError):
                features[f] = None
        else:
            features[f] = None

    if trace_data_path is not None:
        Path(trace_data_path).mkdir(exist_ok=True, parents=True)
        stimuli = evaluator.fitness_protocols["main_protocol"].protocols
        _combo = combo if isinstance(combo, dict) else combo.to_dict()
        hash_id = sha256(json.dumps(_combo).encode()).hexdigest()
        trace_data_path = f"{trace_data_path}/trace_data_{hash_id}.pkl"
        with open(trace_data_path, "wb") as handle:
            pickle.dump([stimuli, responses], handle)

    return {
        "features": json.dumps(features),
        "scores": json.dumps(scores),
        "trace_data": trace_data_path,
    }


def single_feature_evaluation(
    combo,
    access_point=None,
    morphology_path="path",
    stochasticity=False,
    trace_data_path=None,
    timeout=5000,
    threshold_only=False,
    record_ions_and_currents=False,
):
    """Wrapper on top of feature evaluation to have a global timeout"""
    with NestedPool(1, maxtasksperchild=1) as pool:
        res = pool.apply_async(
            _single_feature_evaluation,
            (combo,),
            {
                "access_point": access_point,
                "morphology_path": morphology_path,
                "stochasticity": stochasticity,
                "trace_data_path": trace_data_path,
                "threshold_only": threshold_only,
                "record_ions_and_currents": record_ions_and_currents,
            },
        )
        try:
            out = res.get(timeout=timeout)
        except multiprocessing.TimeoutError:  # pragma: no cover
            print("Timeout computation")
            out = {"features": None, "scores": None, "trace_data": None}
        pool.close()
        pool.join()
    return out


def feature_evaluation(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
    trace_data_path=None,
    stochasticity=False,
    timeout=5000,
    threshold_only=False,
    record_ions_and_currents=False,
):
    """Compute the features and the scores on the combos dataframe.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        continu (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename for the combos sqlite database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed scores and features
    """
    evaluation_function = partial(
        single_feature_evaluation,
        access_point=access_point,
        morphology_path=morphology_path,
        trace_data_path=trace_data_path,
        stochasticity=stochasticity,
        timeout=timeout,
        threshold_only=threshold_only,
        record_ions_and_currents=record_ions_and_currents,
    )

    return (
        evaluate(
            morphs_combos_df.sample(frac=1.0).reset_index(),
            evaluation_function,
            new_columns=[["features", ""], ["scores", ""], ["trace_data", ""]],
            resume=resume,
            parallel_factory=parallel_factory,
            db_url=db_url,
        )
        .sort_values(by="index")
        .set_index("index")
    )


def get_emodel_data(access_point, combo, morphology_path, morph_modifiers):
    """Gather needed emodel data and build cell model for evaluation."""
    access_point.morph_path = Path(combo[morphology_path])

    model_configuration = access_point.get_configuration(combo["emodel"])
    fitness_calculator_configuration = access_point.get_calculator_configuration(combo["emodel"])

    cell = create_cell_model(
        "cell",
        model_configuration=model_configuration,
        morph_modifiers=modifiers.get_synth_modifiers(combo, morph_modifiers=morph_modifiers),
    )

    emodel_params = access_point.final[combo["emodel"]]["params"]
    return cell, fitness_calculator_configuration, emodel_params


def rin_evaluation(
    combo,
    access_point,
    morph_modifiers=None,
    key="rin",
    morphology_path="path",
    stochasticity=False,
    ais_recording=False,
):
    """Evaluating rin protocol as holding -0.02."""
    cell_model, fitness_calculator_configuration, emodel_params = get_emodel_data(
        access_point, combo, morphology_path, deepcopy(morph_modifiers)
    )

    main_protocol, _ = _define_threshold_based_main_protocol(
        fitness_calculator_configuration,
        stochasticity,
        ais_recording=ais_recording,
        strict_holding_bounds=False,
    )

    sim = get_simulator(stochasticity, cell_model)

    try:
        responses = main_protocol.protocols["SearchHoldingCurrent"].run(
            cell_model, emodel_params, sim=sim, timeout=10000
        )
        responses.update(
            main_protocol.protocols["RinProtocol"].run(
                cell_model, emodel_params, responses=responses, sim=sim, timeout=10000
            )
        )
    # pragma: no cover
    except Exception:  # pylint: disable=broad-exception-caught
        print("WARNING: failed rin with:", "".join(traceback.format_exception(*sys.exc_info())))
        return {key: None}

    if responses["bpo_rin"] is None:  # pragma: no cover
        return {key: None}

    if responses["bpo_rin"] < 0:  # pragma: no cover
        print("WARNING: negative rin, reduce holding_voltage to prevent spiking!")
        return {key: None}

    return {key: responses["bpo_rin"]}


def evaluate_rin_no_soma(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
):
    """Compute the input resistance of cell without soma.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin of ais
    """
    key = "rin_no_soma"

    rin_ais_evaluation = partial(
        rin_evaluation,
        access_point=access_point,
        morph_modifiers=[modifiers.remove_soma],
        key=key,
        morphology_path=morphology_path,
        ais_recording=True,
    )
    return evaluate(
        morphs_combos_df,
        rin_ais_evaluation,
        new_columns=[[key, 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def evaluate_soma_rin(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
):
    """Compute the input resistance of the ais (axon).

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin of ais
    """
    key = "rin_soma"

    rin_ais_evaluation = partial(
        rin_evaluation,
        access_point=access_point,
        morph_modifiers=[modifiers.isolate_soma],
        key=key,
        morphology_path=morphology_path,
        ais_recording=False,
    )
    return evaluate(
        morphs_combos_df,
        rin_ais_evaluation,
        new_columns=[[key, 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def evaluate_ais_rin(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
):
    """Compute the input resistance of the ais (axon).

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin of ais
    """
    key = "rin_ais"

    rin_ais_evaluation = partial(
        rin_evaluation,
        access_point=access_point,
        morph_modifiers=[modifiers.isolate_axon],
        key=key,
        morphology_path=morphology_path,
        ais_recording=True,
    )
    return evaluate(
        morphs_combos_df,
        rin_ais_evaluation,
        new_columns=[[key, 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def evaluate_somadend_rin(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
):
    """Compute the input resistance of the soma and dentrites.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        resume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rin or soma+dendrite
    """
    key = "rin_no_ais"
    rin_dendrite_evaluation = partial(
        rin_evaluation,
        access_point=access_point,
        morph_modifiers=[modifiers.remove_axon],
        key=key,
        morphology_path=morphology_path,
    )
    return evaluate(
        morphs_combos_df,
        rin_dendrite_evaluation,
        new_columns=[[key, 0.0]],
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
    )


def evaluate_rho(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
):
    """Compute the input resistances and rho factor.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        rersume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rho axon
    """
    morphs_combos_df = evaluate_soma_rin(
        morphs_combos_df,
        access_point,
        morphology_path=morphology_path,
        resume=resume,
        db_url=db_url,
        parallel_factory=parallel_factory,
    )
    morphs_combos_df = evaluate_rin_no_soma(
        morphs_combos_df,
        access_point,
        morphology_path=morphology_path,
        resume=resume,
        db_url=db_url,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df["rho"] = morphs_combos_df.rin_soma / morphs_combos_df.rin_no_soma
    return morphs_combos_df


def evaluate_rho_axon(
    morphs_combos_df,
    access_point,
    morphology_path="path",
    resume=False,
    db_url=None,
    parallel_factory=None,
):
    """Compute the input resistances and rho factor.

    Args:
        morphs_combos_df (DataFrame): each row reprensents a computation
        access_point (DataAccessPoint): object which contains API to access emodel data
        morphology_path (str): entry from dataframe with morphology paths
        rersume (bool): if True, it will use only compute the empty rows of the database,
            if False, it will ecrase or generate the database
        db_url (str): filename/url for the sql database
        parallel_factory (ParallelFactory): parallel factory instance

    Returns:
        pandas.DataFrame: original combos with computed rho axon
    """
    morphs_combos_df = evaluate_ais_rin(
        morphs_combos_df,
        access_point,
        morphology_path=morphology_path,
        resume=resume,
        db_url=db_url,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df = evaluate_somadend_rin(
        morphs_combos_df,
        access_point,
        morphology_path=morphology_path,
        resume=resume,
        db_url=db_url,
        parallel_factory=parallel_factory,
    )

    morphs_combos_df["rho_axon"] = morphs_combos_df.rin_ais / morphs_combos_df.rin_no_ais
    return morphs_combos_df
