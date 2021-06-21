# Integrating Inference with Integration (MCMSki 2016)

**Lenzerheide, Thursday, 7th January 2016**

<td width="100%" style="position: relative;">
    <img src="../../_static/img/meetings/Lenzerheide_Hochtal.jpg" width="100%">
    <div>
        <small>
            Lenzerheide by
            <a href="https://commons.wikimedia.org/wiki/User:Parpan05">
            Adrian Michael
            </a> /
            <a href="http://creativecommons.org/licenses/by-sa/3.0">
                CC-BY-3.0
            </a>
        </small>
    </div>
</td>

This session will be hosted by [MCMSki](http://www.pages.drexel.edu/~mwl25/mcmskiV/index.html), the joint  meeting of the [Institute of Mathematical Statistics](http://www.imstat.org/) and the [International Society for Bayesian Analysis](http://www.bayesian.org/). The workshop is organised by Michael Osborne, [Chris Oates](http://oates.work/) and [François-Xavier Briol](http://www2.warwick.ac.uk/fac/sci/statistics/staff/research_students/briol/).

## Schedule

The workshop will be held on Thursday, 7th January 2016; the room will be the Activityraum.

* 09:40-10:10: [Simo Särkkä](http://users.aalto.fi/~ssarkka)
* 10:10-10:40: [François-Xavier Briol](http://www2.warwick.ac.uk/fac/sci/statistics/staff/research_students/briol/)
* 10:40-11:10: [Roman Garnett](http://cse.wustl.edu/people/Pages/faculty-bio.aspx?faculty=109)

## Session Theme

Numerical algorithms, such as methods for the numerical solution of integrals, as well as optimization algorithms, can be interpreted as estimation rules. They estimate the value of a latent, intractable quantity – the value of an integral, the solution of a differential equation, the location of an extremum – given the result of tractable computations (“observations”, such as function values of the integrand, evaluations of the differential equation, function values of the gradient of an objective). So these methods perform inference, and are accessible to the formal frameworks of probability theory. They are learning machines.

Taking this observation seriously, a probabilistic numerical method is a numerical algorithm that takes in a probability distribution over its inputs, and returns a probability distribution over its output. Recent research shows that it is in fact possible to directly identify existing numerical methods, including some real classics, with specific probabilistic models.

Interpreting numerical methods as learning algorithms offers various benefits. It can offer insight into the algebraic assumptions inherent in existing methods. As a joint framework for methods developed in separate communities, it allows transfer of knowledge among these areas. But the probabilistic formulation also explicitly provides a richer output than simple convergence bounds. If the probability measure returned by a probabilistic method is well-calibrated, it can be used to monitor, propagate and control the quality of computations.
