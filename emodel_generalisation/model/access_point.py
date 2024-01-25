"""Access point module."""

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

import glob
import json
import logging
from itertools import chain
from pathlib import Path

from emodel_generalisation.model.evaluation import LEGACY_PRE_PROTOCOLS
from emodel_generalisation.model.evaluation import PRE_PROTOCOLS
from emodel_generalisation.model.evaluation import FitnessCalculatorConfiguration
from emodel_generalisation.model.nexus_converter import convert_all_config

# pylint: disable=too-many-lines

logger = logging.getLogger(__name__)

global_parameters = ["v_init", "celsius", "cm", "Ra", "ena", "ek"]
NEURON_BUILTIN_MECHANISMS = ["hh", "pas", "fastpas", "extracellular", "capacitance"]
multiloc_map = {
    "all": ["apical", "basal", "somatic", "axonal"],
    "alldend": ["apical", "basal"],
    "somadend": ["apical", "basal", "somatic"],
    "allnoaxon": ["apical", "basal", "somatic"],
    "somaxon": ["axonal", "somatic"],
    "allact": ["apical", "basal", "somatic", "axonal"],
}


class MorphologyConfiguration:
    """Morphology configuration"""

    def __init__(
        self,
        name=None,
        path=None,
        format=None,  # pylint: disable=redefined-builtin
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Init.

        Args:
            name (str): name of the morphology. If None, the file name from the path will
                be used.
            path (str): path to the morphology file.
            format (str): format of the morphology file, as to be 'asc' or 'swc'. If None,
                the extension of the path will be used.
            seclist_names (list): Optional. Names of the lists of sections
                (e.g: ['somatic', ...]).
            secarray_names (list): Optional. Names of the sections (e.g: ['soma', ...]).
            section_index (list): Optional. Index to a specific section, used for
                non-somatic recordings.
        """

        self.path = None
        if path:
            self.path = str(path)

        if name:
            self.name = name
        elif path:
            self.name = Path(self.path).stem
        else:
            raise TypeError("name or path has to be informed")

        self.format = None
        if format:
            if self.path:
                if format.lower() != path[-3:].lower():
                    raise ValueError("The format does not match the morphology file")
            self.format = format
        elif self.path:
            self.format = path[-3:]

        if self.format and self.format.lower() not in ["asc", "swc"]:
            raise ValueError("The format of the morphology has to be 'asc' or 'swc'.")

        self.seclist_names = seclist_names
        self.secarray_names = secarray_names
        self.section_index = section_index

    def as_dict(self):
        """ """
        return {
            "name": self.name,
            "format": self.format,
            "path": self.path,
            "seclist_names": self.seclist_names,
            "secarray_names": self.secarray_names,
            "section_index": self.section_index,
        }


class DistributionConfiguration:
    """Contains all the information related to the definition and configuration of a parameter
    distribution"""

    def __init__(
        self,
        name,
        function=None,
        parameters=None,
        morphology_dependent_parameters=None,
        soma_ref_location=0.5,
        comment=None,
    ):
        """Init

        Args:
            name (str): name of the distribution.
            function (str): python function of the distribution as a string. Will be executed
                using the python "eval" method. The string needs to include "value" which will be
                replaced by the conductance of the parameter using the present distribution.
                It can also include "distance" if the distribution is parametrized by the distance
                to the soma. Example: "(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}".
            parameters (list of str): names of the parameters that parametrize the above function
                if any. Note that "value" and "distance" do not need to be specified here.
            morphology_dependent_parameters (list of str): unused. To be deprecated.
            soma_ref_location (float): location along the soma used as origin from which to
                compute the distances. Expressed as a fraction (between 0.0 and 1.0).
            comment (str): additional comment or note.
        """

        self.name = name
        self.function = function

        if parameters is None:
            self.parameters = []
        elif isinstance(parameters, str):
            self.parameters = [parameters]
        else:
            self.parameters = parameters

        if morphology_dependent_parameters is None:
            self.morphology_dependent_parameters = []
        elif isinstance(morphology_dependent_parameters, str):
            self.morphology_dependent_parameters = [morphology_dependent_parameters]
        else:
            self.morphology_dependent_parameters = morphology_dependent_parameters

        if soma_ref_location is None:
            soma_ref_location = 0.5
        self.soma_ref_location = soma_ref_location

        self.comment = comment

    def as_dict(self):
        """ """
        distr_dict = {
            "name": self.name,
            "function": self.function,
            "soma_ref_location": self.soma_ref_location,
        }

        for attr in ["parameters", "morphology_dependent_parameters", "comment"]:
            if getattr(self, attr):
                distr_dict[attr] = getattr(self, attr)

        return distr_dict

    def as_legacy_dict(self):
        """ """
        distr_dict = {"fun": self.function}

        if self.parameters:
            distr_dict["parameters"] = self.parameters

        return distr_dict


class ParameterConfiguration:
    """Contains all the information related to the definition and configuration of a parameter"""

    def __init__(self, name, location, value, distribution="uniform", mechanism=None):
        """Init

        Args:
            name (str): name of the parameter. If related to a mechanisms, has to match
                the name of the parameter in the mod file.
            location (str): section of the neuron on which the parameter will be instantiated.
            value (float or list of two floats): if float, set the value of the parameter. If list
                of two floats, sets the upper and lower bound between which the parameter will
                be optimised.
            distribution (str): name of the distribution followed by the parameter (optional).
            mechanism (name): name of the mechanism to which the parameter relates (optional).
        """

        self.name = name
        self.location = location

        self.value = value
        if isinstance(self.value, tuple):
            self.value = list(self.value)
        if isinstance(self.value, list) and len(self.value) == 1:
            self.value = self.value[0]

        self.mechanism = mechanism

        self.distribution = distribution
        if self.distribution is None:
            self.distribution = "uniform"

    @property
    def valid_value(self):
        """ """
        return self.value is not None

    def as_dict(self):
        """ """

        param_dict = {
            "name": self.name,
            "value": self.value,
            "location": self.location,
        }

        if self.distribution and self.distribution != "uniform":
            param_dict["distribution"] = self.distribution

        if self.mechanism:
            param_dict["mechanism"] = self.mechanism

        return param_dict

    def as_legacy_dict(self):
        """ """

        param_dict = {"name": self.name, "val": self.value}

        if self.distribution and self.distribution != "uniform":
            param_dict["dist"] = self.distribution

        return param_dict

    def __eq__(self, other):
        return self.name == other.name and (
            self.name == "all" or other.location == "all" or self.location == other.location
        )


class NeuronModelConfiguration:
    """A neuron model configuration, which includes the model's parameters, distributions,
    mechanisms and a morphology."""

    def __init__(
        self,
        parameters=None,
        mechanisms=None,
        distributions=None,
        morphology=None,
        available_mechanisms=None,
        morph_modifiers=None,
    ):
        """Creates a model configuration, which includes the model parameters, distributions,
        mechanisms and a morphology.

        WARNING: If you create a new NeuronModelConfiguration, do not specify the parameters,
        mechanisms, distributions and morphology here at instantiation. Instead, create an empty
        configuration and then use the class method: add_distribution, add_parameter, etc.
        Example::
            config = NeuronModelConfiguration()
            config.add_parameter(parameter_name, locations, value, mechanism)
            config.morphology = MorphologyConfiguration(morph_name, format=".swc")

        Args:
            parameters (list of dict): contains the description of the parameters of the model
                in the format returned by the method as_dict of the ParameterConfiguration class.
            mechanisms (list of dict): contains the description of the mechanisms of the model
                in the format returned by the method as_dict of the MechanismConfiguration class.
            distributions (list of dict): contains the description of the distributions of the model
                in the format returned by the method as_dict of the DistributionConfiguration class.
            morphology (dict):  contains the description of the morphology of the model in the
                format returned by the method as_dict of the morphology class.
            available_mechanisms (list of MechanismConfiguration): list of the names (
                and optionally versions) of the available mechanisms in the "./mechanisms"
                directory for the local access point or on Nexus for the Nexus access point.
            morph_modifiers (list): List of morphology modifiers. Each modifier has to be
                informed by the path the file containing the modifier and the name of the
                function. E.g: morph_modifiers = [["path_to_module", "name_of_function"], ...].
        """

        if isinstance(parameters, dict):
            self.parameters = [ParameterConfiguration(**parameters)]
        elif isinstance(parameters, list):
            self.parameters = [ParameterConfiguration(**p) for p in parameters]
        else:
            self.parameters = []

        if isinstance(mechanisms, dict):
            self.mechanisms = [MechanismConfiguration(**mechanisms)]
        elif isinstance(mechanisms, list):
            self.mechanisms = [MechanismConfiguration(**m) for m in mechanisms]
        else:
            self.mechanisms = []

        if not distributions:
            self.distributions = []
        elif isinstance(distributions, dict):
            self.distributions = [DistributionConfiguration(**distributions)]
        elif isinstance(distributions, list):
            if isinstance(distributions[0], dict):
                self.distributions = [DistributionConfiguration(**d) for d in distributions]
            elif isinstance(distributions[0], DistributionConfiguration):
                self.distributions = distributions

        for dist in self.distributions:
            if dist.name == "uniform":
                break
        else:
            self.distributions.append(DistributionConfiguration(name="uniform"))

        self.morphology = MorphologyConfiguration(**morphology) if morphology else None

        # TODO: actually use this:
        self.mapping_multilocation = None

        self.available_mechanisms = available_mechanisms

        self.morph_modifiers = morph_modifiers

    @property
    def mechanism_names(self):
        """Returns the names of all the mechanisms used in the model"""

        return {m.name for m in self.mechanisms}

    @property
    def distribution_names(self):
        """Returns the names of all the distributions registered"""

        return {d.name for d in self.distributions}

    @property
    def used_distribution_names(self):
        """Returns the names of all the distributions used in the model"""

        return {p.distribution for p in self.parameters}

    @staticmethod
    def _format_locations(locations):
        """ """

        if locations is None:
            return []
        if isinstance(locations, str):
            return [locations]

        return locations

    def init_from_dict(self, configuration_dict):
        """Instantiate the object from its dictionary form"""

        if "distributions" in configuration_dict:
            for distribution in configuration_dict["distributions"]:
                self.add_distribution(
                    distribution["name"],
                    distribution["function"],
                    distribution.get("parameters", None),
                    distribution.get("soma_ref_location", None),
                )

        if "parameters" in configuration_dict:
            for param in configuration_dict["parameters"]:
                self.add_parameter(
                    param["name"],
                    param["location"],
                    param["value"],
                    param.get("mechanism", None),
                    param.get("dist", None),
                )

        if "mechanisms" in configuration_dict:
            for mechanism in configuration_dict["mechanisms"]:
                self.add_mechanism(
                    mechanism["name"], mechanism["location"], mechanism.get("stochastic", None)
                )

        self.morphology = MorphologyConfiguration(**configuration_dict["morphology"])

    def init_from_legacy_dict(self, parameters, morphology):
        """Instantiate the object from its legacy dictionary form"""

        set_mechanisms = []
        for loc in parameters["mechanisms"]:
            set_mechanisms += parameters["mechanisms"][loc]["mech"]
        set_mechanisms = set(set_mechanisms)

        for distr_name, distr in parameters["distributions"].items():
            self.add_distribution(
                distr_name,
                distr["fun"],
                distr.get("parameters", None),
                distr.get("soma_ref_location", None),
            )

        for location in parameters["parameters"]:
            if location == "__comment":
                continue

            for param in parameters["parameters"][location]:
                mechanism = None

                if (
                    param["name"] not in global_parameters
                    and "distribution" not in location
                    and "_ion" not in param["name"]
                ):
                    mechanism = next((m for m in set_mechanisms if m in param["name"]), None)
                    if mechanism is None:
                        raise ValueError(
                            f"Could not find mechanism associated to parameter {param['name']}"
                        )

                self.add_parameter(
                    param["name"],
                    location,
                    param["val"],
                    mechanism,
                    param.get("dist", None),
                )

        for location in parameters["mechanisms"]:
            for mech in parameters["mechanisms"][location]["mech"]:
                self.add_mechanism(mech, location)

        self.morphology = MorphologyConfiguration(**morphology)

    def add_distribution(
        self,
        distribution_name,
        function,
        parameters=None,
        morphology_dependent_parameters=None,
        soma_ref_location=0.5,
    ):
        """Add a channel distribution to the configuration

        Args:
            distribution_name (str): name of the distribution.
            function (str): python function of the distribution as a string. Will be executed
                using the python "eval" method.
            parameters (list of str): names of the parameters that parametrize the above function
                (no need to include the parameter "distance"). (Optional).
            morphology_dependent_parameters (list of str):
            soma_ref_location (float): location along the soma used as origin
                from which to compute the distances. Expressed as a fraction
                (between 0.0 and 1.0). (Optional).
        """

        tmp_distribution = DistributionConfiguration(
            name=distribution_name,
            function=function,
            parameters=parameters,
            morphology_dependent_parameters=morphology_dependent_parameters,
            soma_ref_location=soma_ref_location,
        )

        self.distributions.append(tmp_distribution)

    def add_parameter(
        self,
        parameter_name,
        locations,
        value,
        mechanism=None,
        distribution_name=None,
        stochastic=None,
        auto_mechanism=True,
    ):
        """Add a parameter to the configuration

        Args:
            parameter_name (str): name of the parameter. If related to a mechanisms, has to match
                the name of the parameter in the mod file.
            locations (str or list of str): sections of the neuron on which these parameters
                will be instantiated.
            value (float or list of two floats): if float, set the value of the parameter. If list
                of two floats, sets the upper and lower bound between which the parameter will
                be optimised.
            mechanism (name): name of the mechanism to which the parameter relates (optional).
            distribution_name (str): name of the distribution followed by the parameter.
                Distributions have to be added before adding the parameters that uses them.
            stochastic (bool): Can the mechanisms to which the parameter relates behave
                stochastically (optional).
            auto_mechanism (bool): if True, will automatically add the mechanism associated to
                the parameter.
        """

        if not locations:
            raise ValueError(
                "Cannot add a parameter without specifying a location. If "
                "global parameter, put 'global'."
            )

        locations = self._format_locations(locations)

        if distribution_name is None or distribution_name == "constant":
            distribution_name = "uniform"

        if distribution_name not in self.distribution_names:
            raise ValueError(
                f"No distribution of name {distribution_name} in the configuration."
                " Please register your distributions first."
            )

        for loc in locations:
            tmp_param = ParameterConfiguration(
                name=parameter_name,
                location=loc,
                value=value,
                distribution=distribution_name,
                mechanism=mechanism,
            )

            if any(p == tmp_param for p in self.parameters):
                logger.debug(
                    "Parameter %s is already at location %s or 'all'. %s",
                    parameter_name,
                    loc,
                    "Only first occurrence will be used.",
                )
            else:
                self.parameters.append(tmp_param)

            if mechanism and auto_mechanism:
                self.add_mechanism(mechanism, loc, stochastic=stochastic)

    def is_mechanism_available(self, mechanism_name, version=None):
        """Is the mechanism part of the mechanisms available"""

        if self.available_mechanisms is not None:
            for mech in self.available_mechanisms:
                if version is not None:
                    if mechanism_name in mech.name and version == mech.version:
                        return True
                elif mechanism_name in mech.name:
                    return True

            return False

        return True

    def add_mechanism(
        self, mechanism_name, locations, stochastic=None, version=None, auto_parameter=False
    ):
        """Add a mechanism to the configuration. This function should rarely be called directly as
         mechanisms are added automatically when using add_parameters. But it might be needed if a
         mechanism is not associated to any parameters.

        Args:
             mechanism_name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
             auto_parameter (bool): if True, will automatically add the parameters of the mechanism
                if they are known.
        """

        locations = self._format_locations(locations)

        if mechanism_name not in NEURON_BUILTIN_MECHANISMS and not self.is_mechanism_available(
            mechanism_name, version
        ):
            raise ValueError(
                f"You are trying to add mechanism {mechanism_name} (version {version}) "
                "but it is not available on Nexus or local."
            )

        for loc in locations:
            if self.available_mechanisms and mechanism_name not in NEURON_BUILTIN_MECHANISMS:
                mechanism_parameters = next(
                    m.parameters for m in self.available_mechanisms if m.name == mechanism_name
                )
            else:
                mechanism_parameters = None

            tmp_mechanism = MechanismConfiguration(
                name=mechanism_name,
                location=loc,
                stochastic=stochastic,
                version=version,
                parameters=mechanism_parameters,
            )

            # Check if mech is not already part of the configuration
            for m in self.mechanisms:
                if m.name == mechanism_name and m.location == loc:
                    return

            # Handle the case where the new mech is a key of the multilocation map
            if loc in multiloc_map:
                tmp_mechanisms = []
                for m in self.mechanisms:
                    if not (m.name == mechanism_name and m.location in multiloc_map[loc]):
                        tmp_mechanisms.append(m)
                self.mechanisms = tmp_mechanisms + [tmp_mechanism]

            # Handle the case where the new mech is a value of the multilocation map
            else:
                for m in self.mechanisms:
                    if m.name == tmp_mechanism.name:
                        if m.location in multiloc_map and loc in multiloc_map[m.location]:
                            return

                self.mechanisms.append(tmp_mechanism)

            if auto_parameter:
                for mechanism_parameter in tmp_mechanism.parameters:
                    self.add_parameter(
                        parameter_name=mechanism_parameter,
                        locations=[tmp_mechanism.location],
                        value=None,
                        mechanism=tmp_mechanism.name,
                        distribution_name="uniform",
                        auto_mechanism=False,
                    )

    def set_parameter_distribution(self, parameter_name, location, distribution_name):
        """Set the distribution of a parameter"""

        if not location:
            raise ValueError("Cannot set a parameter's distribution without specifying a location.")
        locations = self._format_locations(location)

        for loc in locations:
            for p in self.parameters:
                if p.name == parameter_name and p.location == loc:
                    p.distribution = distribution_name

    def set_parameter_value(self, parameter_name, location, value):
        """Set the value of a parameter"""

        if not location:
            raise ValueError("Cannot set a parameter's distribution without specifying a location.")
        locations = self._format_locations(location)

        for loc in locations:
            for p in self.parameters:
                if p.name == parameter_name and p.location == loc:
                    p.value = value

    def select_morphology(
        self,
        morphology_name=None,
        morphology_path=None,
        morphology_format=None,
        seclist_names=None,
        secarray_names=None,
        section_index=None,
    ):
        """Select the morphology on which the neuron model will be based. Its name has to be part
        of the available morphologies.

        Args:
            morphology_name (str): name of the morphology. Optional if morphology_path is informed.
            morphology_path (str): path to the morphology file. If morphology_name is informed, has
                to match the stem of the morphology_path. Optional if morphology_name is informed.
            morphology_format (str): format of the morphology (asc or swc). If morphology_path is
                informed, has to match its suffix. Optional if morphology_format is informed.
            seclist_names (list): Names of the lists of sections (['somatic', ...]) (optional).
            secarray_names (list): names of the sections (['soma', ...]) (optional).
            section_index (int): index to a specific section, used for non-somatic
                recordings (optional).
        """
        self.morphology = MorphologyConfiguration(
            name=morphology_name,
            path=morphology_path,
            format=morphology_format,
            seclist_names=seclist_names,
            secarray_names=secarray_names,
            section_index=section_index,
        )

    def remove_parameter(self, parameter_name, locations=None):
        """Remove a parameter from the configuration. If locations is None or [], the whole
        parameter will be removed. WARNING: that does not remove automatically the mechanism
        which might be still use by other parameter"""

        locations = self._format_locations(locations)

        if locations:
            self.parameters = [
                p
                for p in self.parameters
                if p.name != parameter_name or p.location not in locations
            ]
        else:
            self.parameters = [p for p in self.parameters if p.name != parameter_name]

    def remove_mechanism(self, mechanism_name, locations=None):
        """Remove a mechanism from the configuration and all the associated parameters"""

        locations = self._format_locations(locations)

        if locations:
            self.mechanisms = [
                m
                for m in self.mechanisms
                if m.name != mechanism_name or m.location not in locations
            ]
            self.parameters = [
                p
                for p in self.parameters
                if p.mechanism != mechanism_name or p.location not in locations
            ]
        else:
            self.mechanisms = [m for m in self.mechanisms if m.name != mechanism_name]
            self.parameters = [p for p in self.parameters if p.mechanism != mechanism_name]

    def as_dict(self):
        """Returns the configuration as dict of parameters, mechanisms and
        a list of mechanism names"""

        return {
            "mechanisms": [m.as_dict() for m in self.mechanisms],
            "distributions": [
                d.as_dict() for d in self.distributions if d.name in self.used_distribution_names
            ],
            "parameters": [p.as_dict() for p in self.parameters],
            "morphology": self.morphology.as_dict(),
            "morph_modifiers": self.morph_modifiers,
        }

    def __str__(self):
        """String representation"""

        str_form = "Model Configuration:\n\n"

        str_form += "Mechanisms:\n"
        for m in self.mechanisms:
            str_form += f"   {m.as_dict()}\n"

        str_form += "Distributions:\n"
        for d in self.distributions:
            if d.name in self.used_distribution_names:
                str_form += f"   {d.as_dict()}\n"

        str_form += "Parameters:\n"
        for p in self.parameters:
            str_form += f"   {p.as_dict()}\n"

        str_form += "Morphology:\n"
        str_form += f"   {self.morphology.as_dict()}\n"

        return str_form


class MechanismConfiguration:
    """Contains the information related to the definition and configuration of a mechanism"""

    def __init__(
        self,
        name,
        location,
        stochastic=None,
        version=None,
        parameters=None,
        ion_currents=None,
        nonspecific_currents=None,
        ionic_concentrations=None,
    ):
        """Init

        Args:
             name (str): name of the mechanism.
             locations (str or list of str): sections of the neuron on which this mechanism
                 will be instantiated.
             stochastic (bool): Can the mechanisms behave stochastically (optional).
             version (str): version id of the mod file.
             parameters (list): list of the possible parameter for this mechanism.
             ion_currents (list): list of the ion currents that this mechanism writes.
             nonspecific_currents (list): list of non-specific currents
             ionic_concentrations (list): list of the ionic concentration linked to the ion current
                If None, will be deduced from the ions list.
        """

        self.name = name
        self.location = location
        self.version = version
        self.ion_currents = ion_currents
        self.nonspecific_currents = nonspecific_currents
        self.ionic_concentrations = ionic_concentrations
        if self.ionic_concentrations is None:
            self.ionic_concentrations = []
            if self.ion_currents is not None:
                for ion in self.ion_currents:
                    # remove 'i' in the front and put 'i' at the back to make it a concentration
                    self.ionic_concentrations.append(f"{ion[1:]}i")

        self.stochastic = stochastic
        if self.stochastic is None:
            self.stochastic = "Stoch" in self.name

        if parameters is None:
            self.parameters = {}
        elif isinstance(parameters, str):
            self.parameters = {parameters: [None, None]}
        else:
            self.parameters = parameters

    def get_current(self):
        """Return the ion current names."""
        current = []
        ion_currents = self.ion_currents if self.ion_currents is not None else []
        nonspecific_currents = (
            self.nonspecific_currents if self.nonspecific_currents is not None else []
        )
        for curr in list(chain.from_iterable((ion_currents, nonspecific_currents))):
            current.append(f"{curr}_{self.name}")
        return current

    def as_dict(self):
        """ """
        return {
            "name": self.name,
            "stochastic": self.stochastic,
            "location": self.location,
            "version": self.version,
        }


def get_mechanism_currents(mech_file):
    """Parse the mech mod file to get the mechanism ion and non-specific currents if any."""
    ion_currs = []
    nonspecific_currents = []
    ionic_concentrations = []
    with open(mech_file, "r") as f:
        mod_lines = f.readlines()
    for line in mod_lines:
        if "WRITE " in line:
            ion_var_name = line.split("WRITE ")[1].rstrip("\n").split(" ")[0]
            # ion current case
            if ion_var_name[0] == "i":
                ion_currs.append(ion_var_name)
                ionic_concentrations.append(f"{ion_var_name[1:]}i")
            # internal ionic concentration case
            elif ion_var_name[-1] == "i":
                ionic_concentrations.append(ion_var_name)
        elif "NONSPECIFIC_CURRENT" in line:
            var_name = line.split("NONSPECIFIC_CURRENT ")[1].rstrip("\n").split(" ")[0]
            if var_name[0] == "i":
                nonspecific_currents.append(var_name)

    return ion_currs, nonspecific_currents, ionic_concentrations


def get_mechanism_suffix(mech_file):
    """Parse the mech mod file to get the mechanism suffix."""
    with open(mech_file, "r") as f:
        mod_lines = f.readlines()
    for line in mod_lines:
        if "SUFFIX " in line:
            suffix = line.split("SUFFIX ")[1].rstrip("\n").split(" ")[0]
            return suffix
    raise RuntimeError(f"Could not find SUFFIX in {mech_file}")


class AccessPoint:
    """Custom modifications of BPEM LocalAcessPoint class."""

    def __init__(
        self,
        emodel_dir=None,
        final_path=None,
        recipes_path=None,
        legacy_dir_structure=False,
        with_seeds=False,
        nexus_config=None,
        mech_path="mechanisms",
    ):
        """Init"""
        self.mech_path = mech_path
        if nexus_config is not None:
            if not Path(emodel_dir).exists():
                logger.info("Creating local config folder.")
                convert_all_config(nexus_config, emodel_dir, mech_path=mech_path)
            else:
                logger.info("We found an existing config folder, we will not convert nexus recipe.")
            final_path = Path(emodel_dir) / "final.json"
            recipes_path = Path(emodel_dir) / "recipes.json"

        if emodel_dir is None:
            self.emodel_dir = Path.cwd()
        else:
            self.emodel_dir = Path(emodel_dir)

        self.recipes_path = recipes_path
        self.legacy_dir_structure = legacy_dir_structure
        self.with_seeds = with_seeds

        if final_path is None:
            final_path = self.emodel_dir / "final.json"
        else:
            final_path = Path(final_path)
        self.final = None
        if final_path.exists():
            with open(final_path, "r") as f:
                self.final = json.load(f)
        self.emodels = list(self.final.keys()) if self.final is not None else None
        self.morph_path = None
        self.settings = {}
        self._recipes = None

    @property
    def recipes(self):
        """Cache mechanism to better handle large recipes in non-legacy setting."""
        if self.legacy_dir_structure:
            raise Exception("We cannot use recipes attribute with legacy.")

        if not self._recipes:
            with open(self.recipes_path, "r") as f:
                self._recipes = json.load(f)
        return self._recipes

    def get_recipes(self, emodel):
        """Load the recipes from a json file for an emodel."""
        try:
            _emodel = "_".join(emodel.split("_")[:2]) if self.with_seeds else emodel
            if self.legacy_dir_structure:
                recipes_path = self.emodel_dir / _emodel / "config" / "recipes" / "recipes.json"

                with open(recipes_path, "r") as f:
                    return json.load(f)[_emodel]
            else:
                return self.recipes[_emodel]

        except KeyError:
            _emodel = "_".join(emodel.split("_")[:-1]) if self.with_seeds else emodel
            if self.legacy_dir_structure:
                recipes_path = self.emodel_dir / _emodel / "config" / "recipes" / "recipes.json"
                with open(recipes_path, "r") as f:
                    return json.load(f)[_emodel]
            else:
                return self.recipes[_emodel]

    def get_settings(self, emodel):
        """ """
        settings = self.get_recipes(emodel).get("pipeline_settings", {})
        if self.settings:
            settings.update(self.settings)

        if "morph_modifiers" not in settings:
            settings["morph_modifiers"] = self.get_recipes(emodel).get("morph_modifiers", None)

        return settings

    def get_calculator_configuration(self, emodel, record_ions_and_currents=False):
        """Get the configuration of the fitness calculator (efeatures and protocols)"""
        config_dict = self.get_json(emodel, "features")
        legacy = "efeatures" not in config_dict and "protocols" not in config_dict

        # contains ion currents and ionic concentrations to be recorded
        ion_variables = None
        if record_ions_and_currents:
            ion_currents, ionic_concentrations = self.get_ion_currents_concentrations()
            if ion_currents is not None and ionic_concentrations is not None:
                ion_variables = list(chain.from_iterable((ion_currents, ionic_concentrations)))

        settings = self.get_settings(emodel)
        if legacy:
            efeatures = self.get_json(emodel, "features")
            protocols = self.get_json(emodel, "protocol")

            from_bpe = False
            for protocol_name, protocol in protocols.items():
                if protocol_name in PRE_PROTOCOLS + LEGACY_PRE_PROTOCOLS:
                    continue
                if "stimuli" not in protocol:
                    from_bpe = True
                    break
            configuration = FitnessCalculatorConfiguration(
                name_rmp_protocol=settings.get("name_rmp_protocol", None),
                name_rin_protocol=settings.get("name_Rin_protocol", None),
                stochasticity=settings.get("stochasticity", False),
                ion_variables=ion_variables,
            )

            if from_bpe:
                configuration.init_from_bluepyefe(
                    efeatures,
                    protocols,
                    {},
                    settings.get("threshold_efeature_std", None),
                )
            else:
                configuration.init_from_legacy_dict(
                    efeatures, protocols, settings.get("threshold_efeature_std", None)
                )

        else:
            configuration = FitnessCalculatorConfiguration(
                efeatures=config_dict["efeatures"],
                protocols=config_dict["protocols"],
                name_rmp_protocol=settings.get("name_rmp_protocol", None),
                name_rin_protocol=settings.get("name_Rin_protocol", None),
                stochasticity=settings.get("stochasticity", False),
                ion_variables=ion_variables,
            )

        return configuration

    def get_mechanisms_directory(self):
        """Return the path to the directory containing the mechanisms for the current emodel"""
        if self.emodel_dir:
            mechanisms_directory = self.emodel_dir / "mechanisms"
        else:
            mechanisms_directory = Path("./") / "mechanisms"

        if mechanisms_directory.is_dir():
            return mechanisms_directory.resolve()

        return None

    def get_available_mechanisms(self):
        """Get the list of names of the available mechanisms"""

        mech_dir = self.get_mechanisms_directory()
        if mech_dir is None:
            return None

        available_mechanisms = []
        for mech_file in glob.glob(str(Path(mech_dir) / "*.mod")):
            ion_currents, nonspecific_currents, ion_conc = get_mechanism_currents(mech_file)
            name = get_mechanism_suffix(mech_file)
            available_mechanisms.append(
                MechanismConfiguration(
                    name=name,
                    location=None,
                    ion_currents=ion_currents,
                    nonspecific_currents=nonspecific_currents,
                    ionic_concentrations=ion_conc,
                )
            )

        return available_mechanisms

    def get_json(self, emodel, recipe_entry):
        """Helper function to load a json using path in recipe."""
        json_path = Path(self.get_recipes(emodel)[recipe_entry])

        if self.legacy_dir_structure:
            json_path = self.emodel_dir / "_".join(emodel.split("_")[:2]) / json_path
        elif not json_path.is_absolute():
            json_path = self.emodel_dir / json_path

        with open(json_path, "r") as f:
            return json.load(f)

    def get_morphologies(self, emodel):
        """Get the name and path to the morphologies from the recipes.

        Returns:
            morphology_definition (list): [{'name': morph_name, 'path': 'morph_path'}]. Might
            contain the additional entries "seclist_names" and "secarray_names" if they are
            present in the recipes.
        """
        recipes = self.get_recipes(emodel)

        if self.morph_path is None:
            if isinstance(recipes["morphology"], str):
                morph_file = recipes["morphology"]
            else:
                morph_file = recipes["morphology"][0][1]
            morph_path = Path(recipes["morph_path"]) / morph_file

            if not morph_path.is_absolute():
                morph_path = Path(self.emodel_dir) / morph_path
        else:
            morph_path = Path(self.morph_path)

        morphology_definition = {"name": morph_path.stem, "path": str(morph_path)}
        if "seclist_names" in recipes:
            morphology_definition["seclist_names"] = recipes["seclist_names"]

        if "secarray_names" in recipes:
            morphology_definition["secarray_names"] = recipes["secarray_names"]

        return morphology_definition

    def get_configuration(self, emodel):
        """Get the configuration of the model, including parameters, mechanisms and distributions"""
        configuration = NeuronModelConfiguration(
            available_mechanisms=self.get_available_mechanisms(),
        )
        try:
            parameters = self.get_json(emodel, "parameters")
        except KeyError:
            parameters = self.get_json(emodel, "params")

        if isinstance(parameters["parameters"], dict):
            parameters["parameters"].pop("__comment", None)

        if isinstance(parameters["mechanisms"], dict):
            configuration.init_from_legacy_dict(parameters, self.get_morphologies(emodel))
        else:
            configuration.init_from_dict(parameters)

        return configuration

    def get_ion_currents_concentrations(self):
        """Get all ion currents and ion concentrations.

        Returns:
            tuple containing:

            (list of str): current (ion and non-specific) names for all available mechanisms
            (list of str): ionic concentration names for all available mechanisms
        """
        # pylint: disable=assignment-from-no-return
        mechs = self.get_available_mechanisms()
        if mechs is None:
            return None, None
        ion_currents = list(chain.from_iterable([mech.get_current() for mech in mechs]))
        ionic_concentrations = list(
            chain.from_iterable([mech.ionic_concentrations for mech in mechs])
        )
        # append i_pas which is present by default
        ion_currents.append("i_pas")
        return ion_currents, ionic_concentrations
