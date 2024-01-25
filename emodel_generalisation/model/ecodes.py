"""Ecode module."""

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

import numpy as np
from bluepyopt.ephys.locations import NrnSomaDistanceCompLocation
from bluepyopt.ephys.locations import NrnTrunkSomaDistanceCompLocation
from bluepyopt.ephys.stimuli import Stimulus

logger = logging.getLogger(__name__)


class BPEM_stimulus(Stimulus):
    """Abstract current stimulus"""

    name = ""

    def __init__(self, location):
        """Constructor
        Args:
            total_duration (float): total duration of the stimulus in ms
            location(Location): location of stimulus
        """

        super().__init__()

        self.location = location

        self.iclamp = None
        self.current_vec = None
        self.time_vec = None

    @property
    def stim_start(self):
        """ """
        return 0.0

    @property
    def stim_end(self):
        """ """
        return 0.0

    @property
    def amplitude(self):
        """ """
        return 0.0

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        time_series, current_series = self.generate(dt=0.1)

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = time_series[-1]

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        for t, i in zip(time_series, current_series):
            self.time_vec.append(t)
            self.current_vec.append(i)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def destroy(self, sim=None):  # pylint:disable=W0613
        """Destroy stimulus"""
        self.iclamp = None
        self.time_vec = None
        self.current_vec = None

    def generate(self, dt=0.1):  # pylint:disable=W0613
        """Return current time series"""
        return [], []

    def __str__(self):
        """String representation"""
        return f"{self.name} current played at {self.location}"


class IDrest(BPEM_stimulus):

    """IDrest current stimulus

    .. code-block:: none

              holdi               holdi+amp                holdi
                :                     :                      :
                :                     :                      :
                :           ______________________           :
                :          |                      |          :
                :          |                      |          :
                :          |                      |          :
                :          |                      |          :
        |__________________|                      |______________________
        ^                  ^                      ^                      ^
        :                  :                      :                      :
        :                  :                      :                      :
        t=0                delay                  delay+duration         totduration
    """

    name = "IDrest"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", 200.0)
        self.amp_voltage = kwargs.get("amp_voltage", None)

        self.holding_current = kwargs.get("holding_current", None)
        self.holding_voltage = kwargs.get("holding_voltage", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise TypeError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        self.delay = kwargs.get("delay", 250.0)
        self.duration = kwargs.get("duration", 1350.0)
        self.total_duration = kwargs.get("totduration", 1850.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.delay + self.duration

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.delay)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.delay)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(self.stim_end)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(self.stim_end)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(self.holding_current)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def generate(self, dt=0.1):
        """Return current time series"""

        t = np.arange(0.0, self.total_duration, dt)
        current = np.full(t.shape, self.holding_current, dtype="float64")

        ton_idx = int(self.stim_start / dt)
        toff_idx = int(self.stim_end / dt)

        current[ton_idx:toff_idx] += self.amplitude

        return t, current


class IV(IDrest):

    """IV current stimulus"""

    name = "IV"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        kwargs["thresh_perc"] = kwargs.get("thresh_perc", -40.0)

        kwargs["delay"] = kwargs.get("delay", 250.0)
        kwargs["duration"] = kwargs.get("duration", 3000.0)
        kwargs["totduration"] = kwargs.get("totduration", 3500.0)

        super().__init__(location=location, **kwargs)


class APWaveform(IDrest):

    """APWaveform current stimulus"""

    name = "APWaveform"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        kwargs["thresh_perc"] = kwargs.get("thresh_perc", 220.0)

        kwargs["delay"] = kwargs.get("delay", 250.0)
        kwargs["duration"] = kwargs.get("duration", 50.0)
        kwargs["totduration"] = kwargs.get("totduration", 550.0)

        super().__init__(location=location, **kwargs)


class Comb(BPEM_stimulus):
    # pylint: disable=line-too-long,anomalous-backslash-in-string

    """Comb current stimulus

    .. code-block:: none

              holdi         amp       holdi        amp       holdi           .   .   .
                :            :          :           :          :
                :       ___________     :      ___________     :     ___                  _____
                :      |           |    :     |           |    :    |                          |
                :      |           |    :     |           |    :    |        * n_steps         |
                :      |           |    :     |           |    :    |        .   .   .         |
                :      |           |    :     |           |    :    |                          |
        |______________|           |__________|           |_________|                          |_____
        :              :           :          :           :         :                                ^
        :              :           :          :           :         :                                :
        :              :           :          :           :         :                                :
         <--  delay  --><-duration->           <-duration->         :        .   .   .     totduration
                        <--   inter_delay  --><--  inter_delay   -->

    """  # noqa

    name = "Comb"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
            inter_delay (float): time between each step beginnings in ms
            n_steps (int): number of steps for the stimulus
            amp (float): amplitude of each step(nA)
            delay (float): time at which the first current spike begins (ms)
            duration (float): duration of each step (ms)
            totduration (float): total duration of the whole stimulus (ms)
        """
        self.inter_delay = kwargs.get("inter_delay", 5)
        self.n_steps = kwargs.get("n_steps", 20)
        self.amp = kwargs.get("amp", 40)
        self.delay = kwargs.get("delay", 200.0)
        self.duration = kwargs.get("duration", 0.5)
        self.total_duration = kwargs.get("totduration", 350.0)
        self.holding = 0.0  # hardcoded holding for now (holding_current is modified externally)

        if self.stim_end > self.total_duration:
            raise ValueError(
                "stim_end is larger than total_duration: {self.stim_end} > {self.total_duration})"
            )

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.delay + self.n_steps * self.duration

    @property
    def amplitude(self):
        return self.amp

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()
        self.time_vec.append(self.holding)
        self.current_vec.append(self.holding)

        for step in range(self.n_steps):
            _delay = step * self.inter_delay + self.delay
            self.time_vec.append(_delay)
            self.current_vec.append(self.holding)

            self.time_vec.append(_delay)
            self.current_vec.append(self.amplitude)

            self.time_vec.append(_delay + self.duration)
            self.current_vec.append(self.amplitude)

            self.time_vec.append(_delay + self.duration)
            self.current_vec.append(self.holding)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(self.holding)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def generate(self, dt=0.1):
        """Return current time series"""

        t = np.arange(0.0, self.total_duration, dt)
        current = np.full(t.shape, self.holding, dtype="float64")

        for step in range(self.n_steps):
            _delay = step * self.inter_delay + self.delay
            ton_idx = int(_delay / dt)
            toff_idx = int((_delay + self.duration) / dt)
            current[ton_idx:toff_idx] = self.amplitude

        return t, current


class DendriticStep(IDrest):
    """Step protocol on a dendrite."""

    name = "DendriticStep"

    def __init__(self, location, **kwargs):
        """ """
        direction = kwargs.get("direction", "apical_trunk")
        if direction == "apical_trunk":
            sec_name = kwargs.get("seclist_name", "apical")

            if sec_name != "apical":
                raise ValueError("With direction 'apical_trunk', sec_name must be apical")

            location = NrnTrunkSomaDistanceCompLocation(
                name="dend",
                soma_distance=kwargs["somadistance"],
                sec_index=kwargs.get("sec_index", None),
                seclist_name="apical",
            )
        elif direction == "random":
            location = NrnSomaDistanceCompLocation(
                name="dend",
                soma_distance=kwargs["somadistance"],
                seclist_name=kwargs.get("seclist_name", "apical"),
            )
        else:
            raise ValueError(f"direction keyword {direction} not understood")
        super().__init__(location=location, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Force to have holding current at 0."""
        self.holding_current = 0
        super().instantiate(sim=sim, icell=icell)


class Synaptic(DendriticStep):
    """Ecode to model a synapse with EPSP-like shape.

    A synthetic EPSP shape is defined by the difference of two exponentials, one with a
    rise time (syn_rise) constant, the other with a decay (syn_decay) time constants.
    It is normalized such that the maximum value is parametrized by syn_amp.
    """

    name = "Synaptic"

    def __init__(self, location, **kwargs):
        """Constructor

        Args:
            step_amplitude (float): amplitude (nA)
            step_delay (float): delay (ms)
            step_duration (float): duration (ms)
            location (Location): stimulus Location
            syn_delay (float): start time of synaptic input
            syn_amp (float): maximal amplitude of the synaptic input
            syn_rise (float): rise time constant
            syn_decay (float): decay time constant
        """
        super().__init__(location=None, **kwargs)

        self.syn_delay = kwargs.get("syn_delay", 0.0)
        self.syn_amp = kwargs.get("syn_amp", 0.0)
        self.syn_rise = kwargs.get("syn_rise", 0.5)
        self.syn_decay = kwargs.get("syn_decay", 5.0)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""
        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()
        self.time_vec.append(0.0)
        self.current_vec.append(0)

        self.time_vec.append(self.delay + self.syn_delay)
        self.current_vec.append(0)

        t = np.linspace(0, self.total_duration - self.delay - self.syn_delay, 2000)
        s = np.exp(-t / self.syn_decay) - np.exp(-t / self.syn_rise)
        s = self.syn_amp * s / max(s)

        for _t, _s in zip(t, s):
            self.time_vec.append(self.delay + self.syn_delay + _t)
            self.current_vec.append(_s)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )


class BAC(IDrest):
    """BAC ecode.

    BAC is a combination of a bAP and a synaptic input to generate Ca dendritic spikes.
    """

    def __init__(self, location, **kwargs):
        """Constructor, combination of IDrest and Synaptic ecodes."""
        self.bap = IDrest(location, **kwargs)
        self.epsp = Synaptic(None, **kwargs)

        super().__init__(location=None, **kwargs)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""
        self.bap.holding_current = self.holding_current
        self.bap.threshold_current = self.threshold_current
        self.bap.instantiate(sim=sim, icell=icell)
        self.epsp.instantiate(sim=sim, icell=icell)


class Ramp(BPEM_stimulus):

    """Ramp current stimulus

    .. code-block:: none

            holdi          holdi+amp       holdi
              :                :             :
              :                :             :
              :               /|             :
              :              / |             :
              :             /  |             :
              :            /   |             :
              :           /    |             :
              :          /     |             :
              :         /      |             :
              :        /       |             :
              :       /        |             :
        |___________ /         |__________________________
        ^           ^          ^                          ^
        :           :          :                          :
        :           :          :                          :
        t=0         delay      delay+duration             totduration
    """

    name = "Ramp"

    def __init__(self, location, **kwargs):
        """Constructor
        Args:
            location(Location): location of stimulus
        """

        self.amp = kwargs.get("amp", None)
        self.amp_rel = kwargs.get("thresh_perc", 200.0)

        self.holding_current = kwargs.get("holding_current", None)
        self.threshold_current = None

        if self.amp is None and self.amp_rel is None:
            raise TypeError(f"In stimulus {self.name}, amp and thresh_perc cannot be both None.")

        self.delay = kwargs.get("delay", 250.0)
        self.duration = kwargs.get("duration", 1350.0)
        self.total_duration = kwargs.get("totduration", 1850.0)

        super().__init__(
            location=location,
        )

    @property
    def stim_start(self):
        return self.delay

    @property
    def stim_end(self):
        return self.delay + self.duration

    @property
    def amplitude(self):
        if self.amp_rel is None or self.threshold_current is None:
            return self.amp
        return self.threshold_current * (float(self.amp_rel) / 100.0)

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)

        self.iclamp = sim.neuron.h.IClamp(icomp.x, sec=icomp.sec)
        self.iclamp.dur = self.total_duration

        self.current_vec = sim.neuron.h.Vector()
        self.time_vec = sim.neuron.h.Vector()

        self.time_vec.append(0.0)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.delay)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.stim_end)
        self.current_vec.append(self.holding_current + self.amplitude)

        self.time_vec.append(self.stim_end)
        self.current_vec.append(self.holding_current)

        self.time_vec.append(self.total_duration)
        self.current_vec.append(self.holding_current)

        self.iclamp.delay = 0
        self.current_vec.play(
            self.iclamp._ref_amp,  # pylint:disable=W0212
            self.time_vec,
            1,
            sec=icomp.sec,
        )

    def generate(self, dt=0.1):
        """Return current time series"""

        t = np.arange(0.0, self.total_duration, dt)
        current = np.full(t.shape, self.holding_current, dtype="float64")

        ton_idx = int(self.stim_start / dt)
        toff_idx = int((self.stim_end) / dt)

        current[ton_idx:toff_idx] += np.linspace(0.0, self.amplitude, toff_idx - ton_idx + 1)[:-1]

        return t, current


# The ecode names have to be lower case only, to avoid having to
# define duplicates.
eCodes = {
    "idrest": IDrest,
    "idthres": IDrest,
    "step": IDrest,
    "rinholdcurrent": IDrest,
    "bap": IDrest,
    "iv": IV,
    "spikerec": IDrest,
    "ap_thresh": Ramp,
    "apthresh": Ramp,
    "apwaveform": APWaveform,
    "highfreq": Comb,
    "synaptic": Synaptic,
    "bac": BAC,
    "dendritic": DendriticStep,
    "startnohold": IDrest,
}
