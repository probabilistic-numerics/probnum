"""Interfaces for Bayesian filtering and smoothing."""

from probnum import randprocs


class BayesFiltSmooth:
    """Bayesian filtering and smoothing."""

    def __init__(
        self,
        prior_process: randprocs.MarkovProcess,
    ):
        self.prior_process = prior_process
