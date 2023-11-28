"""Parallel helper."""
import logging
import multiprocessing
import os
from abc import abstractmethod
from collections.abc import Iterator
from functools import partial
from multiprocessing.pool import Pool

import numpy as np

try:
    import dask.distributed
    import dask_mpi

    dask_available = True
except ImportError:  # pragma: no cover
    dask_available = False

try:
    import dask.dataframe as dd  # pylint: disable=ungrouped-imports
    import pandas as pd
    from dask.distributed import progress

    dask_df_available = True
except ImportError:  # pragma: no cover
    dask_df_available = False

try:
    import ipyparallel

    ipyparallel_available = True
except ImportError:  # pragma: no cover
    ipyparallel_available = False


L = logging.getLogger(__name__)


def _func_wrapper(data, func, func_args, func_kwargs):
    """Function wrapper used to pass args and kwargs."""
    return func(data, *func_args, **func_kwargs)


class ParallelFactory:
    """Abstract class that should be subclassed to provide parallel functions."""

    _BATCH_SIZE = "PARALLEL_BATCH_SIZE"
    _CHUNK_SIZE = "PARALLEL_CHUNK_SIZE"

    # pylint: disable=unused-argument
    def __init__(self, batch_size=None, chunk_size=None):
        self.batch_size = batch_size or int(os.getenv(self._BATCH_SIZE, "0")) or None
        L.debug("Using %s=%s", self._BATCH_SIZE, self.batch_size)

        self.chunk_size = batch_size or int(os.getenv(self._CHUNK_SIZE, "0")) or None
        L.debug("Using %s=%s", self._CHUNK_SIZE, self.chunk_size)

        if not hasattr(self, "nb_processes"):
            self.nb_processes = 1

    def __del__(self):
        """Call the shutdown method."""
        self.shutdown()

    @abstractmethod
    def get_mapper(self, batch_size=None, chunk_size=None, **kwargs):
        """Return a mapper function that can be used to execute functions in parallel."""

    def shutdown(self):
        """Can be used to cleanup."""

    def mappable_func(self, func, *args, **kwargs):
        """Can be used to add args and kwargs to a function before calling the mapper."""
        return partial(_func_wrapper, func=func, func_args=args, func_kwargs=kwargs)

    def _with_batches(self, mapper, func, iterable, batch_size=None):
        """Wrapper on mapper function creating batches of iterable to give to mapper.

        The batch_size is an int corresponding to the number of evaluation in each batch.
        """
        if isinstance(iterable, Iterator):
            iterable = list(iterable)

        batch_size = batch_size or self.batch_size
        if batch_size is not None:
            iterables = np.array_split(iterable, len(iterable) // min(batch_size, len(iterable)))
            if not isinstance(iterable, (pd.DataFrame, pd.Series)):
                iterables = [_iterable.tolist() for _iterable in iterables]
        else:
            iterables = [iterable]

        for i, _iterable in enumerate(iterables):
            if len(iterables) > 1:
                L.info("Computing batch %s / %s", i + 1, len(iterables))
            yield from mapper(func, _iterable)

    def _chunksize_to_kwargs(self, chunk_size, kwargs, label="chunk_size"):
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            kwargs[label] = chunk_size


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process."""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):
        """Get daemon flag."""
        return False  # pragma: no cover

    def _set_daemon(self, value):
        """Set daemon flag."""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool."""

    Process = NoDaemonProcess


class SerialFactory(ParallelFactory):
    """Factory that do not work in parallel."""

    def get_mapper(self, batch_size=None, chunk_size=None, **kwargs):
        """Get a map."""

        def _mapper(func, iterable, *func_args, **func_kwargs):
            mapped_func = self.mappable_func(func, *func_args, **func_kwargs)
            return self._with_batches(map, mapped_func, iterable)

        return _mapper


class MultiprocessingFactory(ParallelFactory):
    """Parallel helper class using multiprocessing."""

    _CHUNKSIZE = "PARALLEL_CHUNKSIZE"

    def __init__(self, batch_size=None, chunk_size=None, processes=None, **kwargs):
        """Initialize multiprocessing factory."""
        super().__init__(batch_size, chunk_size)

        self.nb_processes = processes or os.cpu_count()
        self.pool = NestedPool(processes=self.nb_processes, **kwargs)

    def get_mapper(self, batch_size=None, chunk_size=None, **kwargs):
        """Get a NestedPool."""
        self._chunksize_to_kwargs(chunk_size, kwargs, label="chunksize")

        def _mapper(func, iterable, *func_args, **func_kwargs):
            mapped_func = self.mappable_func(func, *func_args, **func_kwargs)
            return self._with_batches(
                partial(self.pool.imap_unordered, **kwargs),
                mapped_func,
                iterable,
            )

        return _mapper


class IPyParallelFactory(ParallelFactory):
    """Parallel helper class using ipyparallel."""

    _IPYTHON_PROFILE = "IPYTHON_PROFILE"

    def __init__(self, batch_size=None, chunk_size=None, profile=None, **kwargs):
        """Initialize the ipyparallel factory."""
        profile = profile or os.getenv(self._IPYTHON_PROFILE, None)
        L.debug("Using %s=%s", self._IPYTHON_PROFILE, profile)
        self.rc = ipyparallel.Client(profile=profile, **kwargs)
        self.nb_processes = len(self.rc.ids)
        self.lview = self.rc.load_balanced_view()
        super().__init__(batch_size, chunk_size)

    def get_mapper(self, batch_size=None, chunk_size=None, **kwargs):
        """Get an ipyparallel mapper using the profile name provided."""
        if "ordered" not in kwargs:  # pragma: no cover
            kwargs["ordered"] = False

        self._chunksize_to_kwargs(chunk_size, kwargs)

        def _mapper(func, iterable, *func_args, **func_kwargs):
            mapped_func = self.mappable_func(func, *func_args, **func_kwargs)
            return self._with_batches(
                partial(self.lview.imap, **kwargs), mapped_func, iterable, batch_size=batch_size
            )

        return _mapper

    def shutdown(self):
        """Remove zmq."""
        try:
            self.rc.close()
        except Exception:  # pragma: no cover ; pylint: disable=broad-except
            pass


class DaskFactory(ParallelFactory):
    """Parallel helper class using dask."""

    _SCHEDULER_PATH = "PARALLEL_DASK_SCHEDULER_PATH"

    def __init__(
        self, batch_size=None, chunk_size=None, scheduler_file=None, address=None, **kwargs
    ):
        """Initialize the dask factory."""
        dask_scheduler_path = scheduler_file or os.getenv(self._SCHEDULER_PATH)
        self.interactive = True
        if dask_scheduler_path:  # pragma: no cover
            L.debug("Connecting dask_mpi with scheduler %s", dask_scheduler_path)
        if address:  # pragma: no cover
            L.debug("Connecting dask_mpi with address %s", address)
        if not dask_scheduler_path and not address:  # pragma: no cover
            self.interactive = False
            # local_directory is a fix for https://github.com/dask/dask-mpi/pull/114
            dask_mpi.initialize(local_directory=None)
            L.debug("Starting dask_mpi...")

        self.client = dask.distributed.Client(
            address=address,
            scheduler_file=dask_scheduler_path,
            **kwargs,
        )

        if self.interactive:
            self.nb_processes = len(self.client.scheduler_info()["workers"])
        else:  # pragma: no cover
            from mpi4py import MPI  # pylint: disable=import-outside-toplevel,import-error

            comm = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
            self.nb_processes = comm.Get_size()

        super().__init__(batch_size, chunk_size)

    def shutdown(self):
        """Close the scheduler and the cluster if it was created by the factory."""
        try:
            self.client.close()
        except Exception:  # pylint: disable=broad-except ; # pragma: no cover
            pass

    def get_mapper(self, batch_size=None, chunk_size=None, **kwargs):
        """Get a Dask mapper."""
        self._chunksize_to_kwargs(chunk_size, kwargs, label="batch_size")

        def _mapper(func, iterable, *func_args, **func_kwargs):
            def _dask_mapper(in_dask_func, iterable):
                futures = self.client.map(in_dask_func, iterable, **kwargs)
                for _future, result in dask.distributed.as_completed(futures, with_results=True):
                    yield result

            mapped_func = self.mappable_func(func, *func_args, **func_kwargs)
            return self._with_batches(_dask_mapper, mapped_func, iterable, batch_size=batch_size)

        return _mapper


class DaskDataFrameFactory(DaskFactory):
    """Parallel helper class using dask.dataframe."""

    _SCHEDULER_PATH = "PARALLEL_DASK_SCHEDULER_PATH"

    def __init__(
        self,
        batch_size=None,
        chunk_size=None,
        scheduler_file=None,
        address=None,
        dask_config=None,
        **kwargs,
    ):
        super().__init__(
            batch_size, chunk_size, scheduler_file=scheduler_file, address=address, **kwargs
        )
        if dask_config is None:  # pragma: no cover
            dask_config = {
                "distributed.worker.use-file-locking": False,
                "distributed.worker.memory.target": False,
                "distributed.worker.memory.spill": False,
                "distributed.worker.memory.pause": 0.8,
                "distributed.worker.memory.terminate": 0.95,
                "distributed.worker.profile.interval": "10000ms",
                "distributed.worker.profile.cycle": "1000000ms",
                "distributed.admin.tick.limit": "1h",
            }

        dask.config.set(dask_config)

    def _with_batches(self, *args, **kwargs):
        """Specific process for batches."""
        for tmp in super()._with_batches(*args, **kwargs):
            if isinstance(tmp, pd.Series):
                tmp = tmp.to_frame()
            yield tmp

    def get_mapper(self, batch_size=None, chunk_size=None, **kwargs):
        """Get a Dask mapper."""
        self._chunksize_to_kwargs(chunk_size, kwargs, label="chunksize")
        if not kwargs.get("chunksize"):
            kwargs["npartitions"] = self.nb_processes or 1

        def _mapper(func, iterable, *func_args, meta, **func_kwargs):
            def _dask_df_mapper(func, iterable):
                df = pd.DataFrame(iterable)
                ddf = dd.from_pandas(df, **kwargs)
                future = ddf.apply(func, meta=meta, axis=1).persist()
                if not os.environ.get("NO_PROGRESS", False):
                    progress(future)
                # Put into a list because of the 'yield from' in _with_batches
                return [future.compute()]

            func = self.mappable_func(func, *func_args, **func_kwargs)
            return self._with_batches(_dask_df_mapper, func, iterable, batch_size=batch_size)

        return _mapper


def init_parallel_factory(parallel_lib, *args, **kwargs):
    """Return the desired instance of the parallel factory.

    The main factories are:

    * None: return a serial mapper (the standard :func:`map` function).
    * multiprocessing: return a mapper using the standard :mod:`multiprocessing`.
    * dask: return a mapper using the :class:`distributed.Client`.
    * ipyparallel: return a mapper using the :mod:`ipyparallel` library.
    """
    parallel_factories = {
        None: SerialFactory,
        "multiprocessing": MultiprocessingFactory,
    }
    if dask_available:  # pragma: no cover
        parallel_factories["dask"] = DaskFactory
    if dask_df_available:  # pragma: no cover
        parallel_factories["dask_dataframe"] = DaskDataFrameFactory
    if ipyparallel_available:  # pragma: no cover
        parallel_factories["ipyparallel"] = IPyParallelFactory

    try:
        parallel_factory = parallel_factories[parallel_lib](*args, **kwargs)
    except KeyError:
        L.critical(
            "The %s factory is not available, maybe the required libraries are not properly "
            "installed.",
            parallel_lib,
        )
        raise
    L.info("Initialized %s factory", parallel_lib)

    return parallel_factory
