# Literature

This page contains a non-exhaustive list of probabilistic numerics research, sorted by problem type.

```{note}
If you would like your publication to be featured in this list, please [open a pull request](https://github.com/probabilistic-numerics/probnum/pulls) on GitHub with the corresponding `bibtex` entry.
```

## General and Foundational
The following papers are often cited as early works on the
idea of uncertainty over the result of deterministic computations.

<!-- {% bibliography --file general %} -->
```{bibliography} bibliography/general.bib
---
all:
---
```

## Quadrature

```{bibliography} bibliography/Quadrature.bib
---
all:
---
```

## Linear Algebra

```{bibliography} bibliography/LinearAlgebra.bib
---
all:
---
```

## Optimization

```{bibliography} bibliography/Optimization.bib
---
all:
---
```

## Ordinary Differential Equations

To avoid a frequent initial confusion for new readers, it may be helpful to
point out that there are two common ways in which probabilistic methods are
used in combination with ordinary differential equations: The "classic" problem
of numerics is to infer the solution to an initial value problem given access
to the differential equation. Below, we term this problem "solving ODEs". The
reverse problem, in some sense, has
also found interest in machine learning: inferring a differential equation from
(noisy) observations of trajectories that are assumed to be governed by this
ODE. Below, this is listed under "inferring ODEs".

### Solving ODEs

```{bibliography} bibliography/ODEs.bib
---
all:
---
```

### Inferring ODEs

```{bibliography} bibliography/ODE_from_path.bib
---
all:
---
```

## Partial Differential Equations

```{bibliography} bibliography/PDEs.bib
---
all:
---
```

<!--
## Approximate Bayesian Computation (ABC)

coming soon


## Applications

coming soon

-->

## Other Related Research

```{bibliography} bibliography/related.bib
---
all:
---
```
