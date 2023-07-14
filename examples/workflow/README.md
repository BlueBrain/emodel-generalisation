To run this example, one should run `run.sh` for local runs, or `run.sh dask` to run with distributed dask.

The worlfow engine is luigi (https://luigi.readthedocs.io/en/stable/), so the parameters of the workflow are in luigi.cfg, and the files created in `out` folder.
To rerun a task, simply remove the corresponding files in `out` folder.
