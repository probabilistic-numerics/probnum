# -*- coding: utf-8 -*-
"""Setup file for probnum.

Use setup.cfg to configure the project.
"""
from setuptools import setup

# TODO (pypa/setuptools#3221): Migrate this to `[project.optional-dependencies]` in
# `pyproject.toml`, once optional dependencies defined there can reference one another
extras_require = dict()
extras_require["jax"] = [
    "jax[cpu]<0.3.5; platform_system!='Windows'",
]
extras_require["torch"] = [
    "torch>=1.11",
]
extras_require["zoo"] = [
    "tqdm>=4.0",
    "requests>=2.0",
] + extras_require["jax"]
extras_require["pls_calib"] = [
    "GPy",
    # GPy can't be imported without matplotlib.
    # This is and should not be a ProbNum dependency.
    "matplotlib",
]
extras_require["full"] = (
    extras_require["jax"] + extras_require["zoo"] + extras_require["pls_calib"]
)


if __name__ == "__main__":
    setup(
        extras_require=extras_require,
    )
