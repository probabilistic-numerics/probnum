Probabilistic Numerics
=======================

Mathematical models used to explain and predict the behaviour of complex systems such as immune cells or the climate
rely heavily on numerical methods. The exponential growth in available data and computing power has revolutionized the
scale of such models. In practice given finite computational resources, this introduces *additional uncertainty* arising
from not running numerical methods to convergence and from observations corrupted by noise.

In a Nutshell
****************

`Probabilistic numerics (PN) <http://probabilistic-numerics.org/>`_ aims to quantify uncertainty arising from
intractable or incomplete numerical computation and from stochastic input. This new paradigm which has emerged at the
intersection of computer science and numerical analysis treats a *numerical problem* as one of *statistical inference*
instead. [#]_ [#]_

The probabilistic viewpoint provides a principled way to encode structural knowledge about a problem. By giving an
explicit role to uncertainty from all sources, in particular from the computation itself, PN gives rise to new
applications beyond the scope of classical methods. For example, it allows consistent propagation of uncertainty
through the entirety of a computational pipeline.

Typical numerical tasks to which PN may be applied include optimization, quadrature, the solution of ordinary and
partial differential equations, and the basic tasks of linear algebra, e.g. solution of linear systems and eigenvalue
problems. Note that the PN approach is different from exploiting randomization in numerical methods, in fact many PN
methods do not rely on sampling.


Value of a Probabilistic Approach
**********************************

As well as offering an enriched reinterpretation of classical methods, the PN approach has several concrete practical
points of value. The probabilistic interpretation of computation

- allows to build customized methods for specific problems with bespoke priors
- formalizes the design of adaptive methods using tools from decision theory
- provides a way of setting parameters of numerical methods via the Bayesian formalism
- expedites the solution of mutually related problems of similar type
- naturally incorporates sources of stochasticity in the computation
- can give structural uncertainty via a probability measure compared to an error estimate

and finally it offers a principled approach of including numerical error in the *propagation of uncertainty through chains of computations*.


.. [#] P. Hennig, M. A. Osborne, and M. Girolami. Probabilistic numerics and uncertainty in computations. Proc. R. Soc. A., 17, 2015.
.. [#] C. J. Oates and T. J. Sullivan. A modern retrospective on probabilistic numerics. Stat. Comput., 29(6):1335–1351, 2019.
