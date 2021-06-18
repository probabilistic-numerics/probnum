# Literature

This page collects literature on all areas of probabilistic
numerics, sorted by problem type. If you would like your publication to be
featured in this list, please do not hesitate to contact us. The fastest way to
get your documents onto the site is to clone our
[github repository](https://github.com/philipphennig/probabilistic-numerics.org),
add your documents to the relevant BibTeX-file in /_bibliography, then either
send us a pull-request, or an email with the updated file (see box on
[the frontpage]({{site.baseurl}}/index.html) for our contacts).

**Quick-jump links:**

* <a href="#General">General and Foundational</a>
* <a href="#Quadrature">Quadrature</a>
* <a href="#Linear">Linear Algebra</a>
* <a href="#Optimization">Optimization</a>
* <a href="#ODEs">Ordinary Differential Equations</a>
* <a href="#PDEs">Partial Differential Equations</a>
* <a href="#Related">Other Related Research</a>

<!-- * <a href="#ABC">Approximate Bayesian Computation</a>
* <a href="#Applications">Applications</a> -->


<h2 id="General">General and Foundational</h2>
The following papers are often cited as early works on the
idea of uncertainty over the result of deterministic computations. Some entries have a "notes" field providing further information about the relevance of the cited work, or pointers to specific results therein.

{% bibliography --file general %}

<h2 id="Quadrature">Quadrature</h2>

{% bibliography --file Quadrature %}

<h2 id="Linear">Linear Algebra</h2>

{% bibliography --file LinearAlgebra %}

<h2 id="Optimization">Optimization</h2>

{% bibliography --file Optimization %}

<h2 id="ODEs">Ordinary Differential Equations</h2>

To avoid a frequent initial confusion for new readers, it may be helpful to
point out that there are two common ways in which probabilistic methods are
used in combination with ordinary differential equations: The "classic" problem
of numerics is to infer the solution to an initial value problem given access
to the differential equation. Below, we term this problem <a
href="#solvingODEs">"solving ODEs"</a>. The reverse problem, in some sense, has
also found interest in machine learning: inferring a differential equation from
(noisy) observations of trajectories that are assumed to be governed by this
ODE. Below, this is listed under <a href="#inferringODEs">"inferring ODEs"</a>.

<h3 id="solvingODEs">Work regarding "solving ODEs"</h3>
{% bibliography --file ODEs %}

<h3 id="inferringODEs">Work regarding "inferring ODEs"</h3>
{% bibliography --file ODE_from_path %}

<h2 id="PDEs">Partial Differential Equations</h2>

{% bibliography --file PDEs %}

<!--
<h2 id="ABC">Approximate Bayesian Computation (ABC)</h2>

coming soon

{% bibliography --file ABC %}

<h2 id="Applications">Applications</h2>

coming soon

{% bibliography --file Applications %}
-->
<h2 id="Related">Other Related Research</h2>

{% bibliography --file related %}
