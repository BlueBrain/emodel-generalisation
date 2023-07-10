"""Workflow utils."""
import os
from pathlib import Path

import luigi
from luigi_tools.target import OutputLocalTarget

from emodel_generalisation.parallel import init_parallel_factory
from emodel_generalisation.model.access_point import AccessPoint


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
