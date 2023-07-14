"""Run MCMC on a simple single compartment model."""
from emodel_generalisation.mcmc import run_several_chains
from emodel_generalisation.parallel import init_parallel_factory

if __name__ == "__main__":
    parallel_factory = init_parallel_factory("multiprocessing")
    # use below to run with dask (and uncomment lines in run.sh)
    # parallel_factory = init_parallel_factory("dask_dataframe")

    run_several_chains(
        proposal_params={"std": 0.04},  # increase std to propose larger jumps
        temperature=1.0,  # increase to explore higher cost values
        n_steps=10000,  # mac number of steps, set high so it will do max
        n_chains=10,  # set to number of cpu in one node
        emodel="all_generic",
        emodel_dir=".",
        recipes_path="config/recipes.json",
        legacy_dir_structure=False,
        run_df_path="run_df.csv",
        results_df_path="chains",
        parallel_lib=parallel_factory,
    )
