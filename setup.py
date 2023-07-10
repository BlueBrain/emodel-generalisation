"""Setup for the emodel-generalisation package."""
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "jsonschema>=3",
    "matplotlib>=2.2.0",
    "morphio>=3.3.4",
    "neurom>=3.2.2",
    "numpy>=1.15.0",
    "pandas>=2.0.3",
    "scipy>=0.13.3",
    "seaborn>=0.12.2",
    "bluepyopt>=1.13.196",
    "luigi-tools>=0.3.3",
    "diameter_synthesis>=0.5.4",
    "scikit-learn>=1.2.2",
    "shap>=0.41.0",
    "xgboost>=1.7.5",
    "pyyaml>=6",
    "datareuse>=0.0.2",
    "ipyparallel>=6.3,<7",
    "dask[dataframe, distributed]>=2021.11",
    "dask-mpi>=2021.11",
    "distributed>=2021.11",
    "sqlalchemy>=1.4.24",
    "sqlalchemy-utils>=0.37.2",
]

doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
]

test_reqs = [
    "pytest>=6",
    "pytest-console-scripts>=1.3",
    "pytest-cov>=3",
    "pytest-html>=2",
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
            "emodel-generalisation=emodel_generalisation.cli:main",
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
