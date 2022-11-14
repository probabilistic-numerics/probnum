"""Fixtures and configuration for doctests."""

import numpy as np

import probnum as pn

import pytest


@pytest.fixture(autouse=True)
def autoimport_packages(doctest_namespace):
    """This fixture 'imports' standard packages automatically in order to avoid
    boilerplate code in doctests."""

    doctest_namespace["pn"] = pn
    doctest_namespace["np"] = np
