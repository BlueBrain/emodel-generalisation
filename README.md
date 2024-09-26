[![DOI](https://zenodo.org/badge/662445885.svg)](https://zenodo.org/badge/latestdoi/662445885)

# emodel-generalisation

Generalisation of neuronal electrical models on a morphological population with Markov Chain Monte-Carlo.

This code accompanies the paper:

[Arnaudon, A., Reva, M., Zbili, M., Markram, H., Van Geit, W., & Kanari, L. (2023). Controlling morpho-electrophysiological variability of neurons with detailed biophysical models. iScience, 2023.](https://www.cell.com/iscience/fulltext/S2589-0042%2823%2902299-X)

## Installation

This code can be installed via [pip](https://pip.pypa.io/en/stable/) from pypi with

```
pip install emodel-generalisation
```

or from github with

```
git clone git@github.com:BlueBrain/emodel-generalisation.git
pip install .
```

## Documentation

The documentation can be found here: https://emodel-generalisation.readthedocs.io/en/latest/

## Code structure

This code contains several modules, the most important are:
* [model](emodel_generalisation/model) contains an adapted version of BlueBrain/BluePyEmodel core functionalities for evaluating electrical models, built on top of BlueBrain/BluePyOpt
* [tasks](emodel_generalisation/tasks) contains the luigi workflows to run MCMC, adapt and generalise electrical model
* [bluecellulab_evaluator](emodel_generalisation/bluecellulab_evaluator.py) contains functions to compute currents with BlueBrain/BlueCelluLab and hoc files of models
* [mcmc](emodel_generalisation/mcmc.py) contains the code to run MCMC sampling of electrical models
* [information](emodel_generalisation/information.py) contains some WIP code to compute information theory measures on sampled electrical models


## Examples

We provide several examples of the main functionalities of the ```emodel-generalisation``` code:
* run MCMC on a simple single compartment model in [examples/mcmc/mcmc_singlecomp](examples/mcmc/mcmc_singlecomp)
* run MCMC on a simple multi-compartment model in [examples/mcmc/mcmc_simple_multicomp](examples/mcmc/mcmc_simple_multicomp)
* run the entire generalisation worklow on a simplified version of the L5PC model of the paper in [examples/workflow](examples/workflow)
* provide all the scripts necessary to reproduce the figures of the paper. For the scripts to run, one has to download the associated dataset on dataverse  with the script ```get_data.sh``` in [examples/paper_figures](examples/paper_figures)


## Citation

When you use the ``emodel-generalisation`` code or method for your research, we ask you to [cite](https://www.cell.com/iscience/fulltext/S2589-0042%2823%2902299-X):

> Arnaudon, A., Reva, M., Zbili, M., Markram, H., Van Geit, W., & Kanari, L. (2023). Controlling morpho-electrophysiological variability of neurons with detailed biophysical models. iScience, 2023.

To get this citation in another format, please use the `Cite this repository` button in the sidebar of the [code's github page](https://github.com/BlueBrain/emodel-generalisation).

## Funding & Acknowledgment

The development of this code was supported by funding to the Blue Brain Project, a research
center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH
Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright 2022-2023 Blue Brain Project/EPFL
