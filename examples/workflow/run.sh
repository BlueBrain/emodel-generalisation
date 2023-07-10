#!/bin/bash -l
#SBATCH --ntasks=1000
##SBATCH --qos=longjob
#SBATCH --time=1-00:00:00       # Time limit
#SBATCH --mem=0
##SBATCH --exclusive
#SBATCH --partition=prod      # Submit to the production 'partition'
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=1
#SBATCH --account=proj82      # your project number
#SBATCH --job-name=mcmc

module purge
module load unstable py-mpi4py
module unload python
#ulimit -n 50000

set -e
# stops Dask workers from spilling to disk
# see https://docs.dask.org/en/latest/setup/hpc.html#local-storage
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
# These two env vars are important to set. With them Dask will stop on error in its master worker (the one that creates Dask's client and calls  `dask_mpi.initialize`).
#export DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD=spawn # spawn may be more stable wrt broken pipe errors, or not...
#export DASK_DISTRIBUTED__WORKER__DAEMON=False
# set to false if file-based locks don't work
# see https://docs.dask.org/en/latest/configuration-reference.html#distributed.worker.use-file-locking
export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
# Reduce dask profile memory usage (it can affect the dashboard precision)
export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile
export DASK_DISTRIBUTED__ADMIN__TICK__LIMIT=1m
#export PARALLEL_BATCH_SIZE=10000
#export PARALLEL_CHUNK_SIZE=20

if [ $1 ]
then
    export PARALLEL_DASK_SCHEDULER_PATH=scheduler.json

    dask-scheduler --scheduler-file ${PARALLEL_DASK_SCHEDULER_PATH} &
    srun dask-mpi  --scheduler-file ${PARALLEL_DASK_SCHEDULER_PATH} --worker-class distributed.Worker --worker-options='{"preload": "import neuron"}' --no-scheduler &
    sleep 30
fi

python -m luigi --module emodel_generalisation.tasks.workflow Run \
    --log-level INFO \
    --local-scheduler
