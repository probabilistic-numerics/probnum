"""Probabilistic numerical methods for solving integrals."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import warnings

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure, LebesgueMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.solvers._bq_state import BQIterInfo, BQState
from probnum.quad.solvers.belief_updates import BQBeliefUpdate, BQStandardBeliefUpdate
from probnum.quad.solvers.initial_designs import InitialDesign, LatinDesign, MCDesign
from probnum.quad.solvers.policies import Policy, RandomPolicy, VanDerCorputPolicy
from probnum.quad.solvers.stopping_criteria import (
    BQStoppingCriterion,
    ImmediateStop,
    IntegralVarianceTolerance,
    MaxNevals,
    RelativeMeanChange,
)
from probnum.quad.typing import DomainLike
from probnum.randprocs.kernels import ExpQuad, Kernel
from probnum.randvars import Normal
from probnum.typing import IntLike

# pylint: disable=too-many-branches, too-complex


class BayesianQuadrature:
    r"""The Bayesian quadrature method.

    Bayesian quadrature solves integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x).

    Parameters
    ----------
    kernel
        The kernel used for the GP model.
    measure
        The integration measure.
    policy
        The policy choosing nodes at which to evaluate the integrand.
    belief_update
        The inference method.
    stopping_criterion
        The criterion that determines convergence.
    initial_design
        The initial design chooses a set of nodes once before the acquisition loop with
        the policy runs.

    See Also
    --------
    bayesquad : Computes the integral using an acquisition policy.
    bayesquad_from_data : Computes the integral :math:`F` using a given dataset of
                          nodes and function evaluations.

    """

    def __init__(
        self,
        kernel: Kernel,
        measure: IntegrationMeasure,
        policy: Optional[Policy],
        belief_update: BQBeliefUpdate,
        stopping_criterion: BQStoppingCriterion,
        initial_design: Optional[InitialDesign],
    ) -> None:
        self.kernel = kernel
        self.measure = measure
        self.policy = policy
        self.belief_update = belief_update
        self.stopping_criterion = stopping_criterion
        self.initial_design = initial_design

    @classmethod
    def from_problem(
        cls,
        input_dim: IntLike,
        kernel: Optional[Kernel] = None,
        measure: Optional[IntegrationMeasure] = None,
        domain: Optional[DomainLike] = None,
        policy: Optional[str] = "bmc",
        initial_design: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> "BayesianQuadrature":

        r"""Creates an instance of this class from a problem description.

        Parameters
        ----------
        input_dim
            The input dimension.
        kernel
            The kernel used for the GP model. Defaults to the ``ExpQuad`` kernel.
        measure
            The integration measure. Defaults to the Lebesgue measure on the ``domain``.
        domain
            The integration bounds. Obsolete if ``measure`` is given.
        policy
            The policy choosing nodes at which to evaluate the integrand.
            Choose ``None`` if you want to integrate from a fixed dataset.
        initial_design
            The initial design samples a set of nodes before the acquisition loop with
            the specified policy runs.
        options
            A dictionary with the following optional solver settings

                scale_estimation : Optional[str]
                    Estimation method to use to compute the scale parameter. Defaults
                    to 'mle'.
                max_evals : Optional[IntLike]
                    Maximum number of evaluations as stopping criterion.
                var_tol : Optional[FloatLike]
                    Variance tolerance as stopping criterion.
                rel_tol : Optional[FloatLike]
                    Relative tolerance as stopping criterion.
                jitter : Optional[FloatLike]
                    Non-negative jitter to numerically stabilise kernel matrix
                    inversion. Defaults to 1e-8.
                batch_size : Optional[IntLike]
                    Batch size used in node acquisition. Defaults to 1.
                num_initial_design_nodes : Optional[IntLike]
                    The number of nodes created by the initial design. Defaults to
                    ``input_dim * 5`` if an initial design is given.

        Returns
        -------
        BayesianQuadrature
            An instance of this class.

        Raises
        ------
        ValueError
            If neither a ``domain`` nor a ``measure`` are given.
        NotImplementedError
            If an unknown ``policy`` is given.
        NotImplementedError
            If an unknown ``initial_design`` is given.
        """

        input_dim = int(input_dim)

        # Set some solver options
        if options is None:
            options = {}

        max_evals = options.get("max_evals", None)
        rel_tol = options.get("rel_tol", None)
        scale_estimation = options.get("scale_estimation", "mle")
        jitter = options.get("jitter", 1.0e-8)
        batch_size = options.get("batch_size", int(1))
        num_initial_design_nodes = options.get(
            "num_initial_design_nodes", int(5 * input_dim)
        )

        # var_tol may be adjusted later if no other stopping condition is given.
        var_tol = options.get("var_tol", None)

        # Set up integration measure
        if domain is None and measure is None:
            raise ValueError(
                "You need to either specify an integration domain or a measure."
            )
        if measure is None:
            measure = LebesgueMeasure(domain=domain, input_dim=input_dim)

        # Select the kernel
        if kernel is None:
            kernel = ExpQuad(input_shape=(input_dim,))

        # Select policy
        if policy is None:
            # If policy is None, this implies that the integration problem is defined
            # through a fixed set of nodes and function evaluations which will not
            # require an acquisition loop. The error handling is done in ``integrate``.
            pass
        elif policy == "bmc":
            policy = RandomPolicy(measure.sample, batch_size=batch_size)
        elif policy == "vdc":
            policy = VanDerCorputPolicy(measure=measure, batch_size=batch_size)
        else:
            raise NotImplementedError(f"The given policy ({policy}) is unknown.")

        # Select the belief updater
        belief_update = BQStandardBeliefUpdate(
            jitter=jitter, scale_estimation=scale_estimation
        )

        # Select initial design
        if initial_design is None:
            pass  # not to raise the exception
        elif initial_design == "mc":
            initial_design = MCDesign(measure, num_initial_design_nodes)
        elif initial_design == "latin":
            initial_design = LatinDesign(measure, num_initial_design_nodes)
        else:
            raise NotImplementedError(
                f"The given initial design ({initial_design}) " f"is unknown."
            )

        # Select stopping criterion: If multiple stopping criteria are given, BQ stops
        # once any criterion is fulfilled (logical `or`).
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

        # If no stopping criteria are given, use some default values.
        if _stopping_criterion is None:
            _stopping_criterion = IntegralVarianceTolerance(var_tol=1e-6) | MaxNevals(
                max_nevals=input_dim * 25  # 25 is an arbitrary value
            )

        # If no policy is given, then the iteration must terminate immediately.
        if policy is None:
            _stopping_criterion = ImmediateStop()

        return cls(
            kernel=kernel,
            measure=measure,
            policy=policy,
            belief_update=belief_update,
            stopping_criterion=_stopping_criterion,
            initial_design=initial_design,
        )

    def bq_iterator(
        self,
        bq_state: BQState,
        info: BQIterInfo,
        fun: Optional[Callable],
        rng: np.random.Generator,
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
        rng
            The random number generator used for random methods.

        Yields
        ------
        new_integral_belief :
            Updated belief about the integral.
        new_bq_state :
            The updated state of the Bayesian quadrature belief.
        new_info :
            The updated state of the iteration.
        """

        while True:

            _has_converged = self.stopping_criterion(bq_state, info)
            info = BQIterInfo.from_stopping_decision(info, has_converged=_has_converged)

            yield bq_state.integral_belief, bq_state, info, rng

            # Have we already converged?
            if _has_converged:
                break

            # Select new nodes via policy
            new_nodes = self.policy(bq_state=bq_state, rng=rng)

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

    def integrate(
        self,
        fun: Optional[Callable],
        nodes: Optional[np.ndarray],
        fun_evals: Optional[np.ndarray],
        rng: Union[IntLike, np.random.Generator] = np.random.default_rng(),
    ) -> Tuple[Normal, BQState, BQIterInfo]:
        """Integrates the function ``fun``.

        ``fun`` may be analytically given, or numerically in terms of ``fun_evals`` at
        fixed nodes. This function calls the generator ``bq_iterator`` until the first
        stopping criterion is met. It immediately stops after processing the initial
        ``nodes`` if ``policy`` is not available.

        Parameters
        ----------
        fun
            Function to be integrated. It needs to accept a shape=(n_eval, input_dim)
            ``np.ndarray`` and return a shape=(n_eval,) ``np.ndarray``.
        nodes
            *shape=(n_eval, input_dim)* -- Optional nodes at which function evaluations
            are available as ``fun_evals`` from start.
        fun_evals
            *shape=(n_eval,)* -- Optional function evaluations at ``nodes`` available
            from the start.
        rng
            The random number generator used for random methods, or a seed.

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
        ValueError
            If ``nodes`` are not given and no policy is present.
        ValueError
            If dimension of ``nodes`` or ``fun_evals`` is incorrect, or if their
            shapes do not match.

        """

        # Get the rng
        if isinstance(rng, IntLike):
            rng = np.random.default_rng(int(rng))

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

        # Setup fixed design
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

        # get initial design nodes
        if self.initial_design is not None:
            initial_design_nodes = self.initial_design(rng)
            initial_design_fun_evals = fun(initial_design_nodes)
            if nodes is not None:
                nodes = np.concatenate((nodes, initial_design_nodes), axis=0)
                fun_evals = np.append(fun_evals, initial_design_fun_evals)
            else:
                nodes = initial_design_nodes
                fun_evals = initial_design_fun_evals

        # set BQ state: This encodes a zero-mean prior.
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

        # set iteration info
        info = BQIterInfo.from_bq_state(bq_state)

        # run loop
        for (_, bq_state, info, rng) in self.bq_iterator(bq_state, info, fun, rng):
            pass

        return bq_state.integral_belief, bq_state, info
