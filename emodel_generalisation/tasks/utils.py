"""Workflow utils."""

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

import os
from pathlib import Path

import luigi
from luigi_tools.target import OutputLocalTarget

from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.parallel import init_parallel_factory


class EmodelAPIConfig(luigi.Config):
    """Configuration of emodel api database."""

    api = luigi.Parameter(default="local")
    emodel_dir = luigi.Parameter(default="configs")
    final_path = luigi.OptionalParameter(default=None)
    emodels = luigi.OptionalListParameter(default=None)
    recipes_path = luigi.OptionalParameter(default=None)
    legacy_dir_structure = luigi.BoolParameter(default=True)


class EmodelAwareTask:
    """Task with loaded access_point."""

    def get_access_point(self, final_path=None, emodel_dir=None, with_seeds=True):
        """Fetch emodel AP."""

        api_config = EmodelAPIConfig()
        if api_config.legacy_dir_structure:
            if emodel_dir is None:
                emodel_dir = (
                    OutputLocalTarget(".").get_prefix()
                    / EmodelLocalTarget(".").get_prefix()
                    / api_config.emodel_dir
                )
            for _emodel in Path(emodel_dir).iterdir():
                if _emodel.is_dir():
                    try:
                        access_point = AccessPoint(
                            emodel_dir=emodel_dir,
                            legacy_dir_structure=True,
                            with_seeds=with_seeds,
                            final_path=final_path,
                        )
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        print(exc)
        else:
            emodel_dir = (
                OutputLocalTarget(".").get_prefix()
                / EmodelLocalTarget(".").get_prefix()
                / api_config.emodel_dir
            )
            access_point = AccessPoint(
                emodel_dir=emodel_dir,
                legacy_dir_structure=False,
                with_seeds=with_seeds,
                final_path=final_path,
                recipes_path=api_config.recipes_path,
            )

        self.emodel_dir = emodel_dir
        return access_point

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)

        self._access_point = None
        self.final_path = None
        self.emodel_dir = None


class ParallelTask:
    """Task with automatic parallel factory detection."""

    def __init__(self, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)
        self._parallel_factory = None

    @property
    def parallel_factory(self):
        """Get parallel factory."""
        if self._parallel_factory is None:
            if os.getenv("IPYTHON_PROFILE"):
                self._parallel_factory = init_parallel_factory("ipyparallel")
            elif os.getenv("PARALLEL_DASK_SCHEDULER_PATH"):
                self._parallel_factory = init_parallel_factory("dask_dataframe")
            else:
                self._parallel_factory = init_parallel_factory("multiprocessing")
        return self._parallel_factory


class EmodelLocalTarget(OutputLocalTarget):
    """Specific target for each emodel."""
