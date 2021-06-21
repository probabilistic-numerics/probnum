# Literature

This page collects literature on all areas of probabilistic
numerics, sorted by problem type. If you would like your publication to be
featured in this list, please open a pull request on GitHub.

## General and Foundational
The following papers are often cited as early works on the
idea of uncertainty over the result of deterministic computations. Some entries have a "notes" field providing further information about the relevance of the cited work, or pointers to specific results therein.

<!-- {% bibliography --file general %} -->
```{bibliography} bibliography/general.bib
---
all:
list: bullet
---
```

## Quadrature

```{bibliography} bibliography/Quadrature.bib
---
all:
list: bullet
---
```

## Linear Algebra

```{bibliography} bibliography/LinearAlgebra.bib
---
all:
list: bullet
---
```

## Optimization

```{bibliography} bibliography/Optimization.bib
---
all:
list: bullet
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
list: bullet
---
```

### Inferring ODEs

```{bibliography} bibliography/ODE_from_path.bib
---
all:
list: bullet
---
```

## Partial Differential Equations

```{bibliography} bibliography/PDEs.bib
---
all:
list: bullet
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
list: bullet
---
```
