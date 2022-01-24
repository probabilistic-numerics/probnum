"""Probabilistic numerical methods for solving integrals."""

import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np

from probnum.quad.solvers.policies import Policy, RandomPolicy
from probnum.quad.solvers.stopping_criteria import (
    BQStoppingCriterion,
    ImmediateStop,
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
)
from probnum.randprocs.kernels import ExpQuad, Kernel
from probnum.randvars import Normal
from probnum.typing import FloatLike, IntLike

from .._integration_measures import IntegrationMeasure, LebesgueMeasure
from ..kernel_embeddings import KernelEmbedding
from .belief_updates import BQBeliefUpdate, BQStandardBeliefUpdate
from .bq_state import BQIterInfo, BQState


class BayesianQuadrature:
    r"""A base class for Bayesian quadrature.

    Bayesian quadrature solves integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x).

    Parameters
    ----------
    kernel :
        The kernel used for the GP model.
    measure :
        The integration measure.
    policy :
        The policy choosing nodes at which to evaluate the integrand.
    belief_update :
        The inference method.
    stopping_criterion :
        The criterion that determines convergence.
    """

    def __init__(
        self,
        kernel: Kernel,
        measure: IntegrationMeasure,
        policy: Policy,
        belief_update: BQBeliefUpdate,
        stopping_criterion: BQStoppingCriterion,
    ) -> None:
        self.kernel = kernel
        self.measure = measure
        self.policy = policy
        self.belief_update = belief_update
        self.stopping_criterion = stopping_criterion

    @classmethod
    def from_problem(
        cls,
        input_dim: int,
        kernel: Optional[Kernel] = None,
        measure: Optional[IntegrationMeasure] = None,
        domain: Optional[
            Union[Tuple[FloatLike, FloatLike], Tuple[np.ndarray, np.ndarray]]
        ] = None,
        policy: Optional[str] = "bmc",
        max_evals: Optional[IntLike] = None,
        var_tol: Optional[FloatLike] = None,
        rel_tol: Optional[FloatLike] = None,
        batch_size: IntLike = 1,
        rng: np.random.Generator = None,
    ) -> "BayesianQuadrature":

        r"""Alternative way to initialize ``Bayesian_Quadrature``

        Parameters
        ----------
        input_dim :
            Input dimension.
        kernel :
            The kernel used for the GP model.
        measure :
            The integration measure.
        domain :
            The integration bounds.
        policy :
            The policy choosing nodes at which to evaluate the integrand.
        max_evals :
            Maximum number of evaluations as stopping criterion.
        var_tol :
            Variance tolerance as stopping criterion.
        rel_tol :
            Relative tolerance as stopping criterion.
        batch_size :
            Batch size used in node acquisition.
        rng :
            The random number generator.

        Returns
        -------
        BayesianQuadrature
            An instance of this class.

        Raises
        ------
        ValueError
            If Bayesian Monte Carlo ('bmc') is selected as ``policy`` and no random
            number generator (``rng``) is given.
        NotImplementedError
            If an unknown ``policy`` is given.
        """
        # Set up integration measure
        if measure is None:
            measure = LebesgueMeasure(domain=domain, input_dim=input_dim)

        # Select the kernel
        if kernel is None:
            kernel = ExpQuad(input_dim=input_dim)

        # Select policy.
        # If policy is not given (None), then the ``integrate'' method will raise an
        # exception if no nodes and function values or a callable are provided there.
        if policy is None:
            pass
        elif policy == "bmc":

            if rng is None:
                errormsg = (
                    "Policy 'bmc' relies on random sampling, "
                    "thus requires a random number generator ('rng')."
                )
                raise ValueError(errormsg)

            policy = RandomPolicy(measure.sample, batch_size=batch_size, rng=rng)

        else:
            raise NotImplementedError(
                "Policies other than random sampling are not available at the moment."
            )

        # Select the belief updater
        belief_update = BQStandardBeliefUpdate()

        # Set stopping criteria: If multiple stopping criteria are given, BQ stops
        # once the first criterion is fulfilled.
        def _stopcrit_or(sc1, sc2):
            if sc1 is None:
                return sc2
            return sc1 | sc2

        _stopping_criterion = None

        if max_evals is not None:
            _stopping_criterion = _stopcrit_or(
                _stopping_criterion, MaxNevals(max_evals)
            )
        if var_tol is not None:
            _stopping_criterion = _stopcrit_or(
                _stopping_criterion, IntegralVarianceTolerance(var_tol)
            )
        if rel_tol is not None:
            _stopping_criterion = _stopcrit_or(
                _stopping_criterion, RelativeMeanChange(rel_tol)
            )

        # If no stopping criteria are given, use some default values
        # (these are arbitrary values)
        if _stopping_criterion is None:
            _stopping_criterion = IntegralVarianceTolerance(var_tol=1e-6) | MaxNevals(
                max_nevals=input_dim * 25
            )

        # If no policy is given, then loop need to be terminated immediately.
        if policy is None:
            _stopping_criterion = ImmediateStop()

        return cls(
            kernel=kernel,
            measure=measure,
            policy=policy,
            belief_update=belief_update,
            stopping_criterion=_stopping_criterion,
        )

    def has_converged(self, bq_state: BQState, info: BQIterInfo) -> bool:
        """Checks if the BQ method has converged.

        Parameters
        ----------
        bq_state
            State of the BQ belief.
        info
            State of the BQ iteration.

        Returns
        -------
        has_converged :
            Whether the solver has converged.
        """

        _has_converged = self.stopping_criterion(bq_state, info)
        if _has_converged:
            info.has_converged = True
            return True
        return False

    def bq_iterator(
        self,
        bq_state: BQState,
        info: Optional[BQIterInfo],
        fun: Optional[Callable],
    ) -> Tuple[Normal, BQState, BQIterInfo]:
        """Generator that implements the iteration of the BQ method.

        This function exposes the state of the BQ method one step at a time while
        running the loop.

        Parameters
        ----------
        bq_state
            State of the Bayesian quadrature methods. Contains the information about
            the problem and the BQ belief.
        info
            The state of the iteration.
        fun
            Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
            ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.

        Yields
        ------
        new_integral_belief :
            Updated belief about the integral.
        new_bq_state :
            The updated state of the Bayesian quadrature belief.
        new_info :
            The updated state of the iteration.
        """

        # Setup iteration info
        if info is None:
            info = BQIterInfo.from_bq_state(bq_state)

        yield bq_state.integral_belief, bq_state, info

        while True:
            # Have we already converged?
            if self.has_converged(bq_state=bq_state, info=info):
                break

            # Select new nodes via policy
            new_nodes = self.policy(bq_state=bq_state)

            # Evaluate the integrand at new nodes
            new_fun_evals = fun(new_nodes)

            # Update the belief about the integrand
            _, bq_state = self.belief_update(
                bq_state=bq_state,
                new_nodes=new_nodes,
                new_fun_evals=new_fun_evals,
            )

            # Update the state of the iteration
            info = BQIterInfo.from_iteration(info=info, dnevals=self.policy.batch_size)

            yield bq_state.integral_belief, bq_state, info

    def integrate(
        self,
        fun: Optional[Callable],
        nodes: Optional[np.ndarray],
        fun_evals: Optional[np.ndarray],
    ) -> Tuple[Normal, BQState, BQIterInfo]:
        """Integrate the function ``fun``.

        ``fun`` may be analytically given, or numerically in terms of ``fun_evals`` at
        fixed nodes. This function calls the generator ``bq_iterator`` until the first
        stopping criterion is met.

        Parameters
        ----------
        fun :
            Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
            ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.
        nodes :
            *shape=(n_eval, input_dim)* -- Optional nodes at which function evaluations
            are available as ``fun_evals`` from start.
        fun_evals :
            *shape=(n_eval,)* -- Optional function evaluations at ``nodes`` available
            from the start.

        Returns
        -------
        integral_belief :
            Posterior belief about the integral.
        bq_state :
            Final state of the Bayesian quadrature method.

        Raises
        ------
        ValueError
            If neither the integrand function (``fun``) nor integrand evaluations
            (``fun_evals``) are given.
        """
        # no policy given: Integrate on fixed dataset.
        if self.policy is None:
            # nodes must be provided if no policy is given.
            if nodes is None:
                raise ValueError("No policy available: Please provide nodes.")

            # Use fun_evals and disregard fun if both are given
            if fun is not None and fun_evals is not None:
                warnings.warn(
                    "No policy available: 'fun_eval' are used instead of 'fun'."
                )
                fun = None

            # override stopping condition as no policy is given.
            self.stopping_criterion = ImmediateStop()

        # Check if integrand function is provided
        if fun is None and fun_evals is None:
            raise ValueError(
                "Please provide an integrand function 'fun' or function values "
                "'fun_evals'."
            )

        # Setup initial design
        if nodes is not None and fun_evals is None:
            fun_evals = fun(nodes)

        # Check if shapes of nodes and function evaluations match
        if fun_evals is not None and fun_evals.ndim != 1:
            raise ValueError(
                f"fun_evals must be one-dimensional " f"({fun_evals.ndim})."
            )
        if nodes is not None and nodes.ndim != 2:
            raise ValueError(f"nodes must be two-dimensional ({nodes.ndim}).")

        if nodes is not None and fun_evals is not None:
            if nodes.shape[0] != fun_evals.shape[0]:
                raise ValueError(
                    f"nodes ({nodes.shape[0]}) and fun_evals "
                    f"({fun_evals.shape[0]}) need to contain the same number "
                    f"of evaluations."
                )

        # Setup BQ state: This encodes a zero-mean prior.
        bq_state = BQState(
            measure=self.measure,
            kernel=self.kernel,
            integral_belief=Normal(
                0.0, KernelEmbedding(self.kernel, self.measure).kernel_variance()
            ),
        )
        if nodes is not None:
            _, bq_state = self.belief_update(
                bq_state=bq_state,
                new_nodes=nodes,
                new_fun_evals=fun_evals,
            )

        info = None
        for (_, bq_state, info) in self.bq_iterator(bq_state, info, fun):
            pass

        return bq_state.integral_belief, bq_state, info
