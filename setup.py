"""
Copyright (c) 2022 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode
or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

"""Setup for the emodel-generalisation package."""
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "matplotlib>=3.6.2",
    "morphio>=3.3.4",
    "neurom>=3.2.2",
    "numpy>=1.23.5",
    "pandas>=1.5.3",
    "scipy>=1.10.0",
    "seaborn>=0.12.2",
    "bluepyopt>=1.13.196",
    "neuron>=8.2.2",
    "morph-tool>=2.9.1",
    "luigi-tools>=0.3.3",
    "diameter_synthesis>=0.5.4",
    "scikit-learn>=1.1.3",
    "shap>=0.41.0",
    "xgboost>=1.7.5,<2",
    "pyyaml>=6",
    "datareuse>=0.0.3",
    "ipyparallel>=6.3,<7",
    "dask[dataframe, distributed]>=2023.3.2",
    "dask-mpi>=2022.4",
    "sqlalchemy>=1.4.24",
    "sqlalchemy-utils>=0.37.2",
    "bluecellulab>=1.7.6",
    "voxcell>=3.1.5",
]

doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
]

test_reqs = [
    "pytest>=7",
    "pytest-console-scripts>=1.3",
    "pytest-cov>=3",
    "pytest-html>=2",
    "pytest-click>=1.1.0",
]

setup(
    name="emodel-generalisation",
    author="Blue Brain Project, EPFL",
    description="Generalisation of neuronal electrical models with MCMC",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://emodel-generalisation.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/emodel-generalisation/issues",
        "Source": "https://github.com/BlueBrain/emodel-generalisation",
    },
    license="GNU General Public License v3.0",
    packages=find_namespace_packages(include=["emodel_generalisation*"]),
    python_requires=">=3.8",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "emodel-generalisation=emodel_generalisation.cli:cli",
        ],
    },
    include_package_data=True,
    classifiers=[
        # TODO: Update to relevant classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
