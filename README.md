# emodel-generalisation

Generalisation of neuronal electrical models on a morphological population with Markov Chain Monte-Carlo.

This code accompanies the pre-print [[1]](#1).

<a id="1">[1]</a> Arnaudon, A., Reva, M., Zbili, M., Markram, H., Van Geit, W., & Kanari, L. (2023). Controlling morpho-electrophysiological variability of neurons with detailed biophysical models. bioRxiv, 2023-04.


## Installation

This code can be installed via [pip](https://pip.pypa.io/en/stable/) (soon on pypi) with

```
pip install emodel-generalisation
```

## Examples

We provide several examples of the main functionalities of ```emodel-generaliastion``` software:
* run MCMC on a simple single compartment model in [hexamples/mcmc/mcmc_singlecomp](examples/mcmc/mcmc_singlecomp)
* run MCMC on a simple multi-compartment model in [examples/mcmc/mcmc_simple_multicomp](examples/mcmc/mcmc_simple_multicomp)
* run the entire generalisation worklow on a simplified version of the L5PC model of the pre-print [[1]](#1) in [examples/workflow](examples/workflow)
* provide all the scripts necessary to reproduce the figures of [[1]](#1). For the scripts to run, one has to download the associated dataset on dataverse  with the script ```get_data.sh``` in [examples/paper_figures](examples/paper_figures)


## Citation

When you use the ``emodel-generalisation`` software or method for your research, we ask you to cite [[1]](#1) associated to this repository (use the `Cite this repository` button on the [main page](https://github.com/BlueBrain/emodel-genealisation) of the code).


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH
Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2022-2023 Blue Brain Project/EPFL
