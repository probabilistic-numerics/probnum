# Development

```{toctree}
---
maxdepth: 1
hidden:
caption: Development
---

development/pull_request
development/developer_guides
development/styleguide
```

Contributions to ProbNum are very welcome! Before getting started make sure to read the following guidelines.

Some suggestions for contributions:

- adding documentation
- adding tutorials
- requesting a feature
- implementing a new feature
- reporting a bug
- fixing a bug
- adding missing tests

If you would like to contribute but are unsure how, then writing tutorials, documentation or working on
[open issues](https://github.com/probabilistic-numerics/probnum/issues) are a good way to start.

We expect all community members to follow the [code of conduct](https://github.com/probabilistic-numerics/probnum/blob/ddada486a0b8bdca0bedfef131344ddd5dad9981/CODE_OF_CONDUCT.md).


## Reporting a Bug
If you find a bug, please report it by opening an issue on the GitHub page, and tag it as a bug.
Reproducibility is key, so if possible provide a working code snippet that reproduces the bug and mention the
Python and ProbNum version you are using.

Please also detail how you found the bug, what you expected to happen, and what happened instead. Have you made any
code changes after which the bug occurred?

Of course, bug fixes via pull request are very welcome as well.

## Requesting a Feature
If you are working with ProbNum, and you are missing a feature, please open an issue and detail what feature you
would like to see. Please also check existing issues in case someone else requested the same or a similar feature.
Detail the intended functionality and use case of the feature.
You can also lay out how you would implement the feature.
If you have a working version, please consider creating a pull request!

## Opening a Pull Request

You implemented an additional feature, a tutorial, tests, or some other enhancement to ProbNum? That's great!
Please consider creating a pull request (PR) with the changes.

If you do so, please check out the
[developer guide](development/pull_request) first.

Before working on a pull request, please check existing open, and recently merged, pull requests to make sure
someone else hasn't addressed the problem already.
If your PR is larger please open an issue first to discuss any significant work - we would hate
for your time to be wasted.

That's it! Thanks for contributing to ProbNum!


# Benchmarking

In ProbNum computational cost with regard to time and memory is measured via a [benchmark suite](https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/).

<div>
	<iframe class="benchmark-preview" src="https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/" allowfullscreen>
		<a href="https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/">ProbNum's Benchmarks</a>
	</iframe>
</div>

Benchmarks are run by [airspeed velocity](https://asv.readthedocs.io/en/stable/) which tracks performance changes over time. You can add a new benchmark yourself by following [this tutorial](https://asv.readthedocs.io/en/stable/writing_benchmarks.html).


```{toctree}
---
maxdepth: 1
hidden:
caption: Benchmarking
---

ProbNum's Benchmarks <https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/>
```

# For Students

```{note}
**Are you a university student and have an interest in ProbNum?**

_University of Tübingen_: If you are interested in a thesis project or student job involving ProbNum, check the listings [here](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/stellen/bachelor-master-available-thesis-topics/) or contact us directly.

_Other Universities_: For students at other universities, we do not offer thesis projects at this time. However, you can still contribute to ProbNum by checking out [open issues on GitHub](https://github.com/probabilistic-numerics/probnum/issues?q=is%3Aopen+is%3Aissue).

```

# ProbNum Team

**Maintainers:** Main developers responsible for ProbNum and the management of the development process.

<div class="authorlist text-center">
	<ul>
		<li>
			<a href="https://github.com/jonathanwenger">
				<img class="avatar" alt="jonathanwenger" src="https://github.com/jonathanwenger.png?v=3&s=96" width="96" height="96" />
				<p>Jonathan Wenger</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/pnkraemer">
				<img class="avatar" alt="pnkraemer" src="https://github.com/pnkraemer.png?v=3&s=96" width="96" height="96" />
				<p>Nicholas Krämer</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/nathanaelbosch">
				<img class="avatar" alt="nathanaelbosch" src="https://github.com/nathanaelbosch.png?v=3&s=96" width="96" height="96" />
				<p>Nathanael Bosch</p>
			</a>
		</li>
	</ul>
</div>
<div style="clear: both"></div>

---

**Code Contributors:** Developers who made substantial additions to ProbNum's codebase or infrastructure.

<div class="authorlist text-center">
	<ul>
		<li>
			<a href="https://github.com/ninaeffenberger">
				<img class="avatar" alt="ninaeffenberger" src="https://github.com/ninaeffenberger.png?v=3&s=96" width="64" height="64" />
				<p>Nina Effenberger</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/alpiges">
				<img class="avatar" alt="alpiges" src="https://github.com/alpiges.png?v=3&s=96" width="64" height="64" />
				<p>Alexandra Gessner</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/coldfix">
				<img class="avatar" alt="coldfix" src="https://github.com/coldfix.png?v=3&s=96" width="64" height="64" />
				<p>Thomas Gläßle</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/tskarvone">
				<img class="avatar" alt="tskarvone" src="https://github.com/tskarvone.png?v=3&s=96" width="64" height="64" />
				<p>Toni Karvonen</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/mmahsereci">
				<img class="avatar" alt="mmahsereci" src="https://github.com/mmahsereci.png?v=3&s=96" width="64" height="64" />
				<p>Maren Mahsereci</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/marvinpfoertner">
				<img class="avatar" alt="marvinpfoertner" src="https://github.com/marvinpfoertner.png?v=3&s=96" width="64" height="64" />
				<p>Marvin Pförtner</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/schmidtjonathan">
				<img class="avatar" alt="schmidtjonathan" src="https://github.com/schmidtjonathan.png?v=3&s=96" width="64" height="64" />
				<p>Jonathan Schmidt</p>
			</a>
		</li>
		<li>
			<a href="https://github.com/jzenn">
				<img class="avatar" alt="jzenn" src="https://github.com/jzenn.png?v=3&s=96" width="64" height="64" />
				<p>Johannes Zenn</p>
			</a>
		</li>
	</ul>
</div>
<div style="clear: both"></div>

---

**Scientific Direction and Funding:** People who give input on the scientific direction or attract and provide funding for research on probabilistic numerics.

<div>
	<a href="https://uni-tuebingen.de/de/134782">
		<img class="avatar" alt="philipphennig" src="https://github.com/philipphennig.png?v=3&s=96" width="48" height="48" />
		Philipp Hennig
	</a>
</div>

---

If you feel like you have made substantial contributions to ProbNum, contact the maintainers of the package.
