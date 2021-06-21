# Probabilistic Numerics (MCQMC 2016)

**Stanford University, Thursday, August 18, 2016**

<td width="100%" style="position: relative;">
    <img src="../../_static/img/meetings/Stanford_Oval_May_2011_panorama.jpg" width="100%">
    <div>
        <small>
            Stanford University by
            <a href="https://commons.wikimedia.org/wiki/User:King_of_Hearts">
            King of Hearts
            </a> /
            <a href="https://creativecommons.org/licenses/by-sa/3.0/">
                CC-BY-SA-3.0
            </a>
        </small>
    </div>
</td>


This workshop is organized by [Mark Girolami](http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/girolami/) and will be hosted by [MCQMC](http://mcqmc2016.stanford.edu/), a biennial meeting on Monte Carlo and quasi-Monte Carlo methods.

Probabilistic Numerics (PN) is an emerging theme of research that straddles the disciplines of mathematical analysis, statistical science and numerical computation. This nascent eld takes it's inspiration from early work in the mid 80s by Diaconis and O'Hagan, who posited that numerical computation should be viewed as a problem of statistical inference. The progress in the intervening years has focused on algorithm design for the probabilistic solution of ordinary differential equations and numerical quadrature, suggesting empirically that an enhanced performance of e.g. numerical quadrature routines can be achieved via an appropriate statistical formulation. However there has been little theoretical analysis to understand the mathematical principles underlying PN that deliver these improvements.

This mini-symposium seeks to bring the area of PN to the MCQMC research community, to present recent mathematical analysis which elucidates the relationship between PN schemes, functional regression and QMC.

## Schedule

The workshop will be held on Thursday, 18 August 15:50-17:50, at Stanford University, Li Ka Shing Center.

### (*15:50-16:20:*) [Mark Girolami](http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/girolami/) - University of Warwick & Alan Turing Institute.

> **Probabilistic Numerical Computation: A New Concept?**
>
> *Abstract:* The vast amounts of data in many different forms becoming available to politicians, policy makers, technologists, and scientists of every hue presents tantalising opportunities for making advances never before considered feasible.<br />
> Yet with these apparent opportunities has come an increase in the complexity of the mathematics required to exploit this data. These sophisticated mathematical representations are much more challenging to analyse, and more and more computationally expensive to evaluate. This is a particularly acute problem for many tasks of interest such as making predictions since these will require the extensive use of numerical solvers for linear algebra, optimization, integration or differential equations. These methods will tend to be slow, due to the complexity of the models, and this will potentially lead to solutions with high levels of uncertainty.<br />
> This talk will introduce our contributions to an emerging area of research defining a nexus of applied mathematics, statistical science and computer science, called “probabilistic numerics”. The aim is to consider numerical problems from a statistical viewpoint, and as such provide numerical methods for which numerical error can be quantified and controlled in a probabilistic manner. This philosophy will be illustrated on problems ranging from predictive policing via crime modelling to computer vision, where probabilistic numerical methods provide a rich and essential quantification of the uncertainty associated with such models and their computation.

### (*16:20-16:50:*) [François-Xavier Briol](http://www2.warwick.ac.uk/fac/sci/statistics/staff/research_students/briol/) - University of Warwick & University of Oxford.

> **Probabilistic Integration: A Role for Statisticians in Numerical Analysis?**
> *Abstract:* One of the current active branches of the field of probabilistic numerics focuses on numerical integration. We will study such a [probabilistic integration](http://www2.warwick.ac.uk/fac/sci/statistics/staff/research_students/briol/probabilistic_integration/) method called Bayesian quadrature, which provides solutions in the form of probability distributions (rather than point estimates) whose structure can provide additional insight into the uncertainty emanating from the finite number of evaluations of the integrand [1]. Strong mathematical guarantees will then be provided in the form of convergence and contraction rates in the number of function evaluations. Finally, we will compare Bayesian quadrature with non-probabilistic methods including Monte Carlo and Quasi-Monte Carlo methods, and illustrate its performance on applications ranging from computer graphics to oil field simulation.<br />
> [1] F-X. Briol, C.J. Oates, M. Girolami, M. A. Osborne and D. Sejdinovic. Probabilistic Integration: A Role for Statisticians in Numerical Analysis? [arXiv:1512.00933](http://arxiv.org/abs/1512.00933), 2015.

### (*16:50-17:20:*) [Chris Oates](http://oates.work/) - University of Technology Sydney.

> **Probabilistic Integration for Intractable Distributions.**
>
> *Abstract:* This talk will build on the theme of Probabilistic Integration, with particular focus on distributions that are intractable, such as posterior distributions in Bayesian statistics. In this context we will discuss a novel approach to estimate posterior expectations using Markov chains [1,2]. This is shown, under regularity conditions, to produce Monte Carlo quadrature rules that converge as
> $\Big|\sum_{i=1}^{n} w_i f(x_i) - \int f(x)p(\mathrm{d}x) \Big| = \mathcal{O}\big(n^{-\frac{1}{2}-\frac{a \wedge b}{2}+\epsilon}\big)$
> at a cost that is cubic in $n$, where $p$ (the posterior density) has $2a+1$ derivatives, the integrand $f$ has $a \wedge b$ derivatives and $\epsilon >0$ can be arbitrarily small.<br />
> [1] C. J. Oates, J. Cockayne, F-X. Briol, and M. Girolami. Convergence Rates for a Class of Estimators Based on Stein's Identity. [arXiv:1603.03220](http://arxiv.org/abs/1603.03220), 2016.<br />
> [2] C. J. Oates, M. Girolami, and N. Chopin. Control Functionals for Monte Carlo integration. Journal of the Royal Statistical Society, Series B, 2017.


### (*17:20-17:50:*) [Jon Cockayne](https://www2.warwick.ac.uk/fac/sci/statistics/staff/research_students/cockayne/) - University of Warwick.

> **Probabilistic meshless methods for partial differential equations and Bayesian inverse problems.**
>
> *Abstract:* Partial differential equations (PDEs) are a challenging class of problems which rarely admit closed-form solutions, forcing us to resort to numerical methods which discretise the problem to construct an approximate solution. We seek to phrase solution of PDEs as a statistical inference problem, and construct probability measures which quantify the epistemic uncertainty in the solution resulting from the discretisation [1]. We explore construction of probability measures for the strong formulation of elliptic PDEs, and the connection between this and "meshless" methods.<br />
> We seek to apply these probability measures in Bayesian inverse problems,
parameter inference problems whose dynamics are often constrained
by a system of PDEs. Sampling from parameter posteriors in such
problems often involves replacing an exact likelihood involving the
unknown solution of the system with an approximate one, in which a
numerical approximation is used. Such approximations have been
shown to produce biased and overconfident posteriors when error
in the forward solver is not tightly controlled. We show how the
uncertainty from a probabilistic forward solver can be propagated into the parameter posteriors, thus permitting the use of coarser
discretisations which still produce valid statistical inferences.<br />
> [1] J. Cockayne, C. J. Oates, T. Sullivan, and M. Girolami. Probabilistic meshless methods for partial differential equations and Bayesian inverse problems. [arXiv:1605.07811](https://arxiv.org/abs/1605.07811), 2016.
