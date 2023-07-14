#!/bin/bash -l
set -e

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

export DASK_DISTRIBUTED__LOGGING__DISTRIBUTED="info"
export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=spawn
export DASK_DISTRIBUTED__WORKER__DAEMON=True

# Reduce dask profile memory usage/leak (see https://github.com/dask/distributed/issues/4091)
export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile

export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=200000ms  # Time for handshake
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=200000ms  # Time for handshake
export DASK_TEMPORARY_DIRECTORY=$TMPDIR

export PARALLEL_DASK_SCHEDULER_PATH=scheduler.json

# uncomment to run with dask on HPC
# dask scheduler --scheduler-file ${PARALLEL_DASK_SCHEDULER_PATH} &
# srun dask-mpi  --scheduler-file ${PARALLEL_DASK_SCHEDULER_PATH} --worker-class distributed.Worker --worker-options='{"preload": "import neuron"}' --no-scheduler &
# sleep 50

python run.py
