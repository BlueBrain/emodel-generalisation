"""Module with overloaded things from bluepyopt."""

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

import logging
import os

import numpy as np
from bluepyopt import ephys
from bluepyopt.ephys.efeatures import eFELFeature

from emodel_generalisation.model.ecodes import eCodes

logger = logging.getLogger(__name__)
# pylint: disable=too-many-lines,import-outside-toplevel,protected-access


class eFELFeatureBPEM(eFELFeature):

    """eFEL feature extra"""

    SERIALIZED_FIELDS = (
        "name",
        "efel_feature_name",
        "recording_names",
        "stim_start",
        "stim_end",
        "exp_mean",
        "exp_std",
        "threshold",
        "comment",
    )

    def __init__(
        self,
        name,
        efel_feature_name=None,
        recording_names=None,
        stim_start=None,
        stim_end=None,
        exp_mean=None,
        exp_std=None,
        threshold=None,
        stimulus_current=None,
        comment="",
        interp_step=None,
        double_settings=None,
        int_settings=None,
        string_settings=None,
    ):
        """Constructor

        Args:
            name (str): name of the eFELFeature object
            efel_feature_name (str): name of the eFeature in the eFEL library
                (ex: 'AP1_peak')
            recording_names (dict): eFEL features can accept several recordings
                as input
            stim_start (float): stimulation start time (ms)
            stim_end (float): stimulation end time (ms)
            exp_mean (float): experimental mean of this eFeature
            exp_std(float): experimental standard deviation of this eFeature
            threshold(float): spike detection threshold (mV)
            comment (str): comment
        """

        super().__init__(
            name,
            efel_feature_name,
            recording_names,
            stim_start,
            stim_end,
            exp_mean,
            exp_std,
            threshold,
            stimulus_current,
            comment,
            interp_step,
            double_settings,
            int_settings,
            string_settings,
            max_score=250.0,
        )

    def calculate_bpo_feature(self, responses):
        """Return internal feature which is directly passed as a response"""

        if self.efel_feature_name not in responses:
            return None

        return responses[self.efel_feature_name]

    def calculate_bpo_score(self, responses):
        """Return score for bpo feature"""

        feature_value = self.calculate_bpo_feature(responses)

        if feature_value is None:
            return self.max_score

        return abs(feature_value - self.exp_mean) / self.exp_std

    def _construct_efel_trace(self, responses):
        """Construct trace that can be passed to eFEL"""

        trace = {}
        if "" not in self.recording_names:
            raise ValueError("eFELFeature: '' needs to be in recording_names")
        for location_name, recording_name in self.recording_names.items():
            if location_name == "":
                postfix = ""
            else:
                postfix = f";{location_name}"

            if recording_name not in responses:
                logger.debug(
                    "Recording named %s not found in responses %s", recording_name, str(responses)
                )
                return None

            if responses[self.recording_names[""]] is None or responses[recording_name] is None:
                return None
            trace[f"T{postfix}"] = responses[self.recording_names[""]]["time"]
            trace[f"V{postfix}"] = responses[recording_name]["voltage"]

            if callable(self.stim_start):
                trace[f"stim_start{postfix}"] = [self.stim_start()]
            else:
                trace[f"stim_start{postfix}"] = [self.stim_start]

            if callable(self.stim_end):
                trace[f"stim_end{postfix}"] = [self.stim_end()]
            else:
                trace[f"stim_end{postfix}"] = [self.stim_end]

        return trace

    def _setup_efel(self):
        """Set up efel before extracting the feature"""

        import efel

        efel.reset()

        if self.threshold is not None:
            efel.setThreshold(self.threshold)

        if self.stimulus_current is not None:
            if callable(self.stimulus_current):
                efel.set_double_setting("stimulus_current", self.stimulus_current())
            else:
                efel.set_double_setting("stimulus_current", self.stimulus_current)

        if self.interp_step is not None:
            efel.set_double_setting("interp_step", self.interp_step)

        if self.double_settings is not None:
            for setting_name, setting_value in self.double_settings.items():
                efel.set_double_setting(setting_name, setting_value)

        if self.int_settings is not None:
            for setting_name, setting_value in self.int_settings.items():
                efel.set_int_setting(setting_name, setting_value)

        if self.string_settings is not None:
            for setting_name, setting_value in self.string_settings.items():
                efel.set_str_setting(setting_name, setting_value)

    def calculate_feature(self, responses, raise_warnings=False):
        """Calculate feature value"""
        if self.efel_feature_name.startswith("bpo_"):
            feature_values = np.array([self.calculate_bpo_feature(responses)])

        else:
            efel_trace = self._construct_efel_trace(responses)

            if efel_trace is None:
                feature_values = None
            else:
                self._setup_efel()
                logger.debug("Amplitude for %s: %s", self.name, self.stimulus_current)
                import efel

                values = efel.getFeatureValues(
                    [efel_trace], [self.efel_feature_name], raise_warnings=raise_warnings
                )

                feature_values = values[0][self.efel_feature_name]

                efel.reset()

        logger.debug("Calculated values for %s: %s", self.name, str(feature_values))
        return feature_values

    def calculate_score(self, responses, trace_check=False):
        """Calculate the score"""

        if self.efel_feature_name.startswith("bpo_"):
            score = self.calculate_bpo_score(responses)

        elif self.exp_mean is None:
            score = 0

        else:
            feature_values = self.calculate_feature(responses)

            # ensures no burst is a valid feature
            if feature_values is None and self.efel_feature_name == "burst_number":
                feature_values = np.array([0])

            if feature_values is None or len(feature_values) == 0:
                score = self.max_score
            else:
                score = (
                    np.sum(np.fabs(feature_values - self.exp_mean))
                    / self.exp_std
                    / len(feature_values)
                )
                logger.debug("Calculated score for %s: %f", self.name, score)

            score = np.min([score, self.max_score])

        if score is None or np.isnan(score):
            return self.max_score

        return score


def get_i_membrane(isection):
    """Look for i_membrane in a location."""
    raw_dict = isection.psection()["density_mechs"]
    if "extracellular" in raw_dict:
        if "i_membrane" in raw_dict["extracellular"]:
            return ["i_membrane"]
    return []


def get_loc_currents(isection):
    """Get all overall currents available in a location."""
    local_overall_currs = set()

    # ion overall current & concentration
    ions = isection.psection()["ions"]
    for _, ion in ions.items():
        for var in ion.keys():
            # current should have 'i' at the beginning (e.g. ik, ica, ina, ...)
            if var[0] == "i":
                local_overall_currs.add(var)

    return local_overall_currs


def get_loc_varlist(isection):
    """Get all possible variables in a location."""
    local_varlist = []

    # currents etc.
    raw_dict = isection.psection()["density_mechs"]
    for channel, vars_ in raw_dict.items():
        for var in vars_.keys():
            local_varlist.append("_".join((var, channel)))

    local_varlist.append("v")

    return local_varlist


def check_recordings(recordings, icell, sim):
    """Returns a list of valid recordings (where the variable is in the location)."""

    new_recs = []  # to return
    varlists = {}  # keep varlists to avoid re-computing them every time

    for rec in recordings:
        # get section from location
        try:
            seg = rec.location.instantiate(sim=sim, icell=icell)
        except ephys.locations.EPhysLocInstantiateException:
            continue
        sec = seg.sec
        section_key = str(sec)

        # get list of variables available in the section
        if section_key in varlists:
            local_varlist = varlists[section_key]
        else:
            local_varlist = (
                get_loc_varlist(sec)
                + list(get_loc_ions(sec))
                + list(get_loc_currents(sec))
                + get_i_membrane(sec)
            )
            varlists[section_key] = local_varlist

        # keep recording if its variable is available in its location
        if rec.variable in local_varlist:
            rec.checked = True
            new_recs.append(rec)

    return new_recs


class BPEMProtocol(ephys.protocols.SweepProtocol):

    """Base protocol"""

    def __init__(
        self, name=None, stimulus=None, recordings=None, cvode_active=None, stochasticity=False
    ):
        """Constructor

        Args:
            name (str): name of this object
            stimulus (Stimulus): stimulus objects
            recordings (list of Recordings): Recording objects used in the
                protocol
            cvode_active (bool): whether to use variable time step
            stochasticity (bool): turns on or off the channels that can be
                stochastic
        """

        super().__init__(
            name=name,
            stimuli=[stimulus],
            recordings=recordings,
            cvode_active=cvode_active,
            deterministic=not stochasticity,
        )

        self.stimulus = stimulus

        self.features = []

    def instantiate(self, sim=None, cell_model=None):
        """Check recordings, then instantiate."""
        if not all(rec.checked for rec in self.recordings):
            self.recordings = check_recordings(self.recordings, cell_model.icell, sim)

        super().instantiate(sim, cell_model)

    def stim_start(self):
        """Time stimulus starts"""
        return self.stimulus.stim_start

    def stim_end(self):
        """ """
        return self.stimulus.stim_end

    def amplitude(self):
        """ """
        return self.stimulus.amplitude

    def run(  # pylint: disable=arguments-differ, arguments-renamed
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""

        param_values = {} if param_values is None else param_values
        responses = {} if responses is None else responses

        if os.environ.get("DISABLE_CVODE", False):
            self.cvode_active = False

        return super().run(
            cell_model=cell_model,
            param_values=param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )


def get_loc_ions(isection):
    """Get all ion concentrations available in a location."""
    local_overall_ions = set()

    # ion overall current & concentration
    ions = isection.psection()["ions"]
    for _, ion in ions.items():
        for var in ion.keys():
            # concentration should have 'i' at the end (e.g. ki, cai, nai, ...)
            if var[-1] == "i":
                local_overall_ions.add(var)

    return local_overall_ions


class LooseDtRecordingCustom(ephys.recordings.CompRecording):
    """Recording that can be checked, but that do not records at fixed dt."""

    def __init__(self, name=None, location=None, variable="v"):
        """Constructor.

        Args:
            name (str): name of this object
            location (Location): location in the model of the recording
            variable (str): which variable to record from (e.g. 'v')
        """
        super().__init__(name=name, location=location, variable=variable)

        # important to turn current densities into currents
        self.segment_area = None
        # important to detect ion concentration variable
        self.local_ion_list = None
        self.checked = False

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording."""
        logger.debug("Adding compartment recording of %s at %s", self.variable, self.location)

        self.varvector = sim.neuron.h.Vector()
        seg = self.location.instantiate(sim=sim, icell=icell)
        self.varvector.record(getattr(seg, f"_ref_{self.variable}"))

        self.segment_area = seg.area()
        self.local_ion_list = get_loc_ions(seg.sec)

        self.tvector = sim.neuron.h.Vector()
        self.tvector.record(sim.neuron.h._ref_t)  # pylint: disable=W0212

        self.instantiated = True

    @property
    def response(self):
        """Return recording response. Turn current densities into currents."""

        if not self.instantiated:
            return None

        # do not modify voltage or ion concentration
        if self.variable == "v" or self.variable in self.local_ion_list:
            return ephys.responses.TimeVoltageResponse(
                self.name, self.tvector.to_python(), self.varvector.to_python()
            )

        # ionic current: turn mA/cm2 (*um2) into pA
        return ephys.responses.TimeVoltageResponse(
            self.name,
            self.tvector.to_python(),
            np.array(self.varvector.to_python()) * self.segment_area * 10.0,
        )


class FixedDtRecordingCustom(LooseDtRecordingCustom):
    """Recording that can be checked, with recording every 0.1 ms."""

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording."""
        logger.debug("Adding compartment recording of %s at %s", self.variable, self.location)

        self.varvector = sim.neuron.h.Vector()
        seg = self.location.instantiate(sim=sim, icell=icell)
        self.varvector.record(getattr(seg, f"_ref_{self.variable}"), 0.1)

        self.segment_area = seg.area()
        self.local_ion_list = get_loc_ions(seg.sec)

        self.tvector = sim.neuron.h.Vector()
        self.tvector.record(sim.neuron.h._ref_t, 0.1)  # pylint: disable=W0212

        self.instantiated = True


class ResponseDependencies:

    """To add to a protocol to specify that it depends on the responses of other protocols"""

    def __init__(self, dependencies=None):
        """Constructor

        Args:
            dependencies (dict): dictionary of dependencies of the form
                {self_attribute_name: [protocol_name, response_name]}.
        """

        self.dependencies = {} if dependencies is None else dependencies

    def return_none_responses(self):
        """ """
        # pylint: disable=inconsistent-return-statements
        raise NotImplementedError()

    def set_attribute(self, attribute, value):
        """Set an attribute of the class based on the name of the attribute. Also handles
        the case where the name is of the form: attribute.sub_attribute"""

        if "." in attribute:
            obj2 = getattr(self, attribute.split(".")[0])
            setattr(obj2, attribute.split(".")[1], value)
        else:
            setattr(self, attribute, value)

    def set_dependencies(self, responses):
        """ """
        for attribute_name, dep in self.dependencies.items():
            if responses.get(dep[1], None) is None:
                logger.debug("Dependency %s missing", dep[1])
                return False
            self.set_attribute(attribute_name, responses[dep[1]])
        return True

    def _run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        raise NotImplementedError("The run code of the sub-classes goes here!")

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""

        param_values = {} if param_values is None else param_values
        responses = {} if responses is None else responses

        if not self.set_dependencies(responses):
            return self.return_none_responses()

        return self._run(cell_model, param_values, sim, isolate, timeout)


class ProtocolWithDependencies(BPEMProtocol, ResponseDependencies):

    """To add to a protocol to specify that it depends on the responses of other protocols"""

    def __init__(
        self,
        dependencies=None,
        name=None,
        stimulus=None,
        recordings=None,
        cvode_active=None,
        stochasticity=False,
    ):
        """Constructor

        Args:
            dependencies (dict): dictionary of dependencies of the form
                {self_attribute_name: [protocol_name, response_name]}.
        """

        ResponseDependencies.__init__(self, dependencies)
        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochasticity=stochasticity,
        )

    def return_none_responses(self):
        """ """
        # pylint: disable=inconsistent-return-statements
        raise NotImplementedError()

    def _run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        return BPEMProtocol.run(self, cell_model, param_values, sim, isolate, timeout, responses)


class ReboundBurst(BPEMProtocol):

    """Protocol for rebound bursting of thalamic cells."""

    def __init__(
        self, name=None, stimulus=None, recordings=None, cvode_active=None, stochasticity=False
    ):
        """Constructor"""

        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochasticity=stochasticity,
        )

    def get_holding_current_from_voltage(
        self,
        voltage,
        cell_model,
        param_values=None,
        sim=None,
        isolate=None,
        timeout=None,
        responses=None,
    ):
        """Run bisection search to get holding current that match holding voltage."""
        # maybe not the best, this is copied from normal holding search
        target_voltage = eFELFeatureBPEM(
            "target_voltage",
            efel_feature_name="steady_state_voltage_stimend",
            recording_names={"": "target_voltage.soma.v"},
            exp_mean=voltage,
            exp_std=1,
            threshold=-20.0,
        )
        hold_prot = SearchHoldingCurrent(
            "target_voltage",
            self.stimulus.location,
            target_voltage=target_voltage,
            stimulus_duration=2000.0,
        )
        if self.stimulus.holding_current is not None:
            hold_prot.stimulus.delay = 1000
            hold_prot.stimulus.holding_current = self.stimulus.holding_current

        return hold_prot.run(cell_model, param_values, sim, isolate, timeout, responses)[
            "bpo_holding_current"
        ]

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""

        if self.stimulus.holding_voltage is not None:
            self.stimulus.holding_current = self.get_holding_current_from_voltage(
                self.stimulus.holding_voltage,
                cell_model,
                param_values,
                sim,
                isolate,
                timeout,
                responses,
            )

        if self.stimulus.amp_voltage is not None:
            self.stimulus.amp = self.get_holding_current_from_voltage(
                self.stimulus.amp_voltage,
                cell_model,
                param_values,
                sim,
                isolate,
                timeout,
                responses,
            )
        return BPEMProtocol.run(self, cell_model, param_values, sim, isolate, timeout, responses)


class ThresholdBasedProtocol(ProtocolWithDependencies):

    """Protocol having rheobase-rescaling capabilities. When using ThresholdBasedProtocol,
    the current amplitude and step amplitude of the stimulus will be ignored and replaced by
    values obtained from the holding current and rheobase of the cell model respectively."""

    def __init__(
        self, name=None, stimulus=None, recordings=None, cvode_active=None, stochasticity=False
    ):
        """Constructor"""

        dependencies = {
            "stimulus.holding_current": ["SearchHoldingCurrent", "bpo_holding_current"],
            "stimulus.threshold_current": ["SearchThresholdCurrent", "bpo_threshold_current"],
        }

        ProtocolWithDependencies.__init__(
            self,
            dependencies=dependencies,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=cvode_active,
            stochasticity=stochasticity,
        )

    def return_none_responses(self):
        return {k.name: None for k in self.recordings}

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        return ResponseDependencies.run(
            self, cell_model, param_values, sim, isolate, timeout, responses
        )


class RMPProtocol(BPEMProtocol):

    """Protocol consisting of a step of amplitude zero"""

    def __init__(self, name, location, target_voltage, stimulus_duration=500.0):
        """Constructor"""

        stimulus_definition = {
            "delay": 0.0,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": stimulus_duration,
            "holding_current": 0.0,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]

        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.stimulus_duration = stimulus_duration

        self.target_voltage = target_voltage
        self.target_voltage.stim_start = stimulus_duration - 100.0
        self.target_voltage.stim_end = stimulus_duration
        self.target_voltage.stimulus_current = 0.0

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Compute the RMP"""

        response = BPEMProtocol.run(
            self,
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
            responses=responses,
        )
        if responses is None or response[self.recording_name] is None:
            return {self.recording_name: None, "bpo_rmp": None}

        bpo_rmp = self.target_voltage.calculate_feature(response)
        response["bpo_rmp"] = bpo_rmp if bpo_rmp is None else bpo_rmp[0]

        return response


class RinProtocol(ProtocolWithDependencies):

    """Protocol used to find the input resistance of a model"""

    def __init__(
        self,
        name,
        location,
        target_rin,
        amp=-0.02,
        stimulus_delay=500.0,
        stimulus_duration=500.0,
        totduration=1000.0,
    ):
        """Constructor"""

        stimulus_definition = {
            "delay": stimulus_delay,
            "amp": amp,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": totduration,
            "holding_current": None,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]

        dependencies = {"stimulus.holding_current": ["SearchHoldingCurrent", "bpo_holding_current"]}

        ProtocolWithDependencies.__init__(
            self,
            dependencies=dependencies,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.target_rin = target_rin
        self.target_rin.stim_start = stimulus_delay
        self.target_rin.stim_end = stimulus_delay + stimulus_duration
        self.target_rin.stimulus_current = amp

    def return_none_responses(self):
        return {self.recording_name: None, "bpo_rin": None}

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Compute the Rin"""

        response = ResponseDependencies.run(
            self,
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
            responses=responses,
        )

        bpo_rin = self.target_rin.calculate_feature(response)
        response["bpo_rin"] = bpo_rin if bpo_rin is None else bpo_rin[0]

        return response


class SearchHoldingCurrent(BPEMProtocol):
    """Protocol used to find the holding current of a model"""

    def __init__(
        self,
        name,
        location,
        target_voltage=None,
        voltage_precision=0.1,
        stimulus_duration=500.0,
        upper_bound=0.2,
        lower_bound=-0.2,
        strict_bounds=True,
        max_depth=7,
        no_spikes=True,
    ):
        """Constructor

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_voltage (EFeature): target for the voltage at holding_current
            voltage_precision (float): accuracy for holding voltage, in mV, to stop the search
            stimulus_duration (float): length of the protocol
            upper_bound (float): upper bound for the holding current, in pA
            lower_bound (float): lower bound for the holding current, in pA
            strict_bounds (bool): to adaptively enlarge bounds if current is outside
            max_depth (int): maximum depth for the binary search
            no_spikes (bool): if True, the holding current will only be considered valid if there
                are no spikes at holding.
        """

        stimulus_definition = {
            # if we put larger, and we have spikes before end of delay, Spikecount will fail as it
            # takes entire traces
            "delay": 0.0,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": stimulus_duration,
            "holding_current": 0.0,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]

        BPEMProtocol.__init__(
            self,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.voltage_precision = voltage_precision
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.strict_bounds = strict_bounds
        self.no_spikes = no_spikes

        self.target_voltage = target_voltage
        self.holding_voltage = self.target_voltage.exp_mean

        self.target_voltage.stim_start = stimulus_duration - 100.0
        self.target_voltage.stim_end = stimulus_duration
        self.target_voltage.stimulus_current = 0.0

        self.max_depth = max_depth

        # start/end are not used by this feature
        self.spike_feature = ephys.efeatures.eFELFeature(
            name=f"{name}.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"{name}.{location.name}.v"},
            stim_start=0,
            stim_end=1000,
            exp_mean=1,
            exp_std=0.1,
        )

    def get_voltage_base(
        self, holding_current, cell_model, param_values, sim, isolate, timeout=None
    ):
        """Calculate voltage base for a certain holding current"""

        self.stimuli[0].amp = holding_current
        response = BPEMProtocol.run(
            self, cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
        )

        if response is None or response[self.recording_name] is None:
            return None

        if self.no_spikes:
            n_spikes = self.spike_feature.calculate_feature(response)
            if n_spikes is not None and n_spikes > 0:
                return None

        voltage_base = self.target_voltage.calculate_feature(response)
        if voltage_base is None:
            return None
        return voltage_base[0]

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""
        if not self.strict_bounds:
            # first readjust the bounds if needed
            voltage_min = 1e10
            while voltage_min > self.target_voltage.exp_mean:
                voltage_min = self.get_voltage_base(
                    holding_current=self.lower_bound,
                    cell_model=cell_model,
                    param_values=param_values,
                    sim=sim,
                    isolate=isolate,
                    timeout=timeout,
                )
                if voltage_min is None:
                    voltage_min = 1e10

                if voltage_min > self.target_voltage.exp_mean:
                    self.lower_bound -= 0.2

            voltage_max = -1e10
            while voltage_max < self.target_voltage.exp_mean:
                voltage_max = self.get_voltage_base(
                    holding_current=self.upper_bound,
                    cell_model=cell_model,
                    param_values=param_values,
                    sim=sim,
                    isolate=isolate,
                    timeout=timeout,
                )
                if voltage_max is None:
                    # if we spike, we let it pass to the search
                    voltage_max = 1e10

                elif voltage_max < self.target_voltage.exp_mean:
                    self.upper_bound += 0.2

        response = {
            "bpo_holding_current": self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                upper_bound=self.upper_bound,
                lower_bound=self.lower_bound,
                timeout=timeout,
            )
        }

        if response["bpo_holding_current"] is None:
            return response

        response.update(
            BPEMProtocol.run(
                self, cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout
            )
        )

        return response

    def bisection_search(
        self,
        cell_model,
        param_values,
        sim,
        isolate,
        upper_bound,
        lower_bound,
        timeout=None,
        depth=0,
    ):
        """Do bisection search to find holding current"""
        mid_bound = (upper_bound + lower_bound) * 0.5
        voltage = self.get_voltage_base(
            holding_current=mid_bound,
            cell_model=cell_model,
            param_values=param_values,
            sim=sim,
            isolate=isolate,
            timeout=timeout,
        )
        # if we don't converge fast enough, we stop and return lower bound, which will not spike
        if depth > self.max_depth:
            logging.debug(
                "Exiting search due to reaching max_depth. The required voltage precision "
                "was not reached."
            )
            return lower_bound

        if voltage is not None and abs(voltage - self.holding_voltage) < self.voltage_precision:
            logger.debug("Depth of holding search: %s", depth)
            return mid_bound

        # if voltage is None, it means we spike at mid_bound, so we try with lower side
        if voltage is None or voltage > self.holding_voltage:
            return self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                lower_bound=lower_bound,
                upper_bound=mid_bound,
                depth=depth + 1,
            )

        return self.bisection_search(
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            lower_bound=mid_bound,
            upper_bound=upper_bound,
            depth=depth + 1,
        )


class SearchThresholdCurrent(ProtocolWithDependencies):

    """Protocol used to find the threshold current (rheobase) of a model"""

    def __init__(
        self,
        name,
        location,
        target_threshold=None,
        current_precision=1e-2,
        stimulus_delay=500.0,
        stimulus_duration=2000.0,
        stimulus_totduration=3000.0,
        max_threshold_voltage=-30,
        spikecount_timeout=50,
        max_depth=10,
        no_spikes=True,
    ):
        """Constructor.

        Args:
            name (str): name of this object
            location (Location): location on which to perform the search (
                usually the soma).
            target_threshold (Efeature): target for the threshold_current
            current_precision (float): size of search interval in current to stop the search
            stimulus_delay (float): delay before the beginning of the step
                used to create the protocol
            stimulus_duration (float): duration of the step used to create the
                protocol
            stimulus_totduration (float): total duration of the protocol
            max_threshold_voltage (float): maximum voltage used as upper
                bound in the threshold current search
            spikecount_timeout (float): timeout for spikecount computation, if timeout is reached,
                we set spikecount=2 as if many spikes were present, to speed up bisection search.
            max_depth (int): maximum depth for the binary search
            no_spikes (bool): if True, will check that the holding current (lower bound) does not
                trigger spikes.
        """

        dependencies = {
            "stimulus.holding_current": ["SearchHoldingCurrent", "bpo_holding_current"],
            "rin": ["RinProtocol", "bpo_rin"],
            "rmp": ["RMPProtocol", "bpo_rmp"],
        }

        stimulus_definition = {
            "delay": stimulus_delay,
            "amp": 0.0,
            "thresh_perc": None,
            "duration": stimulus_duration,
            "totduration": stimulus_totduration,
            "holding_current": None,
        }

        self.recording_name = f"{name}.{location.name}.v"
        stimulus = eCodes["step"](location=location, **stimulus_definition)
        recordings = [
            LooseDtRecordingCustom(name=self.recording_name, location=location, variable="v")
        ]

        super().__init__(
            dependencies=dependencies,
            name=name,
            stimulus=stimulus,
            recordings=recordings,
            cvode_active=True,
            stochasticity=False,
        )

        self.rin = None
        self.rmp = None

        self.target_threshold = target_threshold
        self.max_threshold_voltage = max_threshold_voltage
        self.current_precision = current_precision
        self.no_spikes = no_spikes
        self.max_depth = max_depth

        # start/end are not used by this feature
        self.spike_feature = ephys.efeatures.eFELFeature(
            name=f"{name}.Spikecount",
            efel_feature_name="Spikecount",
            recording_names={"": f"{name}.{location.name}.v"},
            stim_start=0,
            stim_end=1000,
            exp_mean=1,
            exp_std=0.1,
        )
        self.spikecount_timeout = spikecount_timeout

    def return_none_responses(self):
        return {"bpo_threshold_current": None, self.recording_name: None}

    def _get_spikecount(self, current, cell_model, param_values, sim, isolate):
        """Get spikecount at a given current."""
        self.stimulus.amp = current

        response = self._run(
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            timeout=self.spikecount_timeout,
        )
        if response[self.recording_name] is None:
            logger.debug("Trace computation for threshold timed out at %s", self.spikecount_timeout)
            return 2

        return self.spike_feature.calculate_feature(response)

    def run(
        self, cell_model, param_values=None, sim=None, isolate=None, timeout=None, responses=None
    ):
        """Run protocol"""

        if not self.set_dependencies(responses):
            return self.return_none_responses()

        lower_bound, upper_bound = self.define_search_bounds(
            cell_model, param_values, sim, isolate, responses
        )
        if lower_bound is None or upper_bound is None:
            logger.debug("Threshold search bounds are not good")
            return {"bpo_threshold_current": None}

        threshold = self.bisection_search(
            cell_model,
            param_values,
            sim,
            isolate,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            timeout=timeout,
        )

        response = {"bpo_threshold_current": threshold}
        if threshold is None:
            return response

        self.stimulus.amp = threshold
        response.update(
            self._run(cell_model, param_values, sim=sim, isolate=isolate, timeout=timeout)
        )

        return response

    def max_threshold_current(self):
        """Find the current necessary to get to max_threshold_voltage"""
        max_threshold_current = (self.max_threshold_voltage - self.rmp) / self.rin
        max_threshold_current = np.min([max_threshold_current, 2.0])
        logger.debug("Max threshold current: %.6g", max_threshold_current)
        return max_threshold_current

    def define_search_bounds(self, cell_model, param_values, sim, isolate, responses):
        """Define the bounds and check their validity"""

        upper_bound = self.max_threshold_current()
        spikecount = self._get_spikecount(upper_bound, cell_model, param_values, sim, isolate)
        if spikecount == 0:
            return None, None

        lower_bound = responses["bpo_holding_current"]

        if lower_bound > upper_bound:
            return None, None

        return lower_bound, upper_bound

    def bisection_search(
        self,
        cell_model,
        param_values,
        sim,
        isolate,
        upper_bound,
        lower_bound,
        timeout=None,
        depth=0,
    ):
        """Do bisection search to find threshold current."""
        mid_bound = (upper_bound + lower_bound) * 0.5
        spikecount = self._get_spikecount(mid_bound, cell_model, param_values, sim, isolate)
        if abs(lower_bound - upper_bound) < self.current_precision:
            logger.debug("Depth of threshold search: %s", depth)
            return upper_bound

        if depth > self.max_depth:
            return upper_bound

        if spikecount == 0:
            return self.bisection_search(
                cell_model,
                param_values,
                sim=sim,
                isolate=isolate,
                lower_bound=mid_bound,
                upper_bound=upper_bound,
                timeout=timeout,
                depth=depth + 1,
            )
        return self.bisection_search(
            cell_model,
            param_values,
            sim=sim,
            isolate=isolate,
            lower_bound=lower_bound,
            upper_bound=mid_bound,
            timeout=timeout,
            depth=depth + 1,
        )


class ProtocolRunner(ephys.protocols.Protocol):

    """Meta-protocol in charge of running the other protocols in the correct order"""

    def __init__(self, protocols, name="ProtocolRunner"):
        """Initialize the protocol runner

        Args:
            protocols (dict): Dictionary of protocols to run
            name (str): Name of the current protocol runner
        """

        super().__init__(name=name)

        self.protocols = protocols
        self.execution_order = self.compute_execution_order()

    def _add_to_execution_order(self, protocol, execution_order, before_index=None):
        """Recursively adds protocols to the execution order while making sure that their
        dependencies are added before them. Warning: Does not solve all types of dependency graph.
        """

        if protocol.name not in execution_order:
            if before_index is None:
                execution_order.append(protocol.name)
            else:
                execution_order.insert(before_index, protocol.name)

        if hasattr(protocol, "dependencies"):
            for dep in protocol.dependencies.values():
                if dep[0] not in execution_order:
                    self._add_to_execution_order(
                        self.protocols[dep[0]],
                        execution_order,
                        before_index=execution_order.index(protocol.name),
                    )

    def compute_execution_order(self):
        """Compute the execution order of the protocols by taking into account their dependencies"""

        execution_order = []

        for protocol in self.protocols.values():
            self._add_to_execution_order(protocol, execution_order)

        return execution_order

    def run(self, cell_model, param_values, sim=None, isolate=None, timeout=None):
        """Run protocol"""

        responses = {}
        cell_model.freeze(param_values)

        for protocol_name in self.execution_order:
            logger.debug("Computing protocol %s", protocol_name)
            new_responses = self.protocols[protocol_name].run(
                cell_model,
                param_values={},
                sim=sim,
                isolate=isolate,
                timeout=timeout,
                responses=responses,
            )

            if new_responses is None or any(v is None for v in new_responses.values()):
                logger.debug("None in responses, exiting evaluation")
                break

            responses.update(new_responses)

        cell_model.unfreeze(param_values.keys())
        return responses
