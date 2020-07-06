# Implementing a Probabilistic Numerical Method

Probabilistic numerical methods in ProbNum follow the dogma of having `RandomVariable`s as input and `RandomVariable`s 
as output. Hence their signature should look like this:
```python
randvar_out = probnum_method(problem, randvar_in, **kwargs)
```
where `problem` defines the numerical problem to solve, `randvar_in` is the input random variable encoding prior 
information and `**kwargs` are keyword arguments that influence the behaviour 
of the solver (e.g. control termination). If your method has a classic analogue in `numpy` or `scipy`, make sure the 
signatures match as closely as possible. This enables PN methods to be used as drop-in replacements for classic 
numerical routines.

## Interface and Implementation
Each method in ProbNum has an interface called by the user defined through a function similar to the one below

```python
def my_probnum_method(problem, randvar_in, type="vanilla", **kwargs):
    """
    Solve a numerical problem using a probabilistic numerical method.

    This probabilistic numerical method solves the problem xyz, by ...

    Parameters
    ----------
    problem : Problem, shape=(n,n)
        Arguments defining the numerical problem to be solved.
    randvar_in : RandomVariable, shape=(n,)
        Input random variable encoding prior information about the problem.
    type : str
        Variant of probabilistic numerical method to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         advanced              ``adv``
        ====================  ===========

    Returns
    -------
    randvar_out : RandomVariable, shape=(n,)
        Output random variable with posterior distribution over the quantity to be estimated.

    Raises
    ------
    ValueError
        Input shapes do not match.
    """
    # Choose method
    if type == "vanilla":
        pnmethod = VanillaPnMethod(problem, randvar_in)
    elif type == "wasabi":
        pnmethod = AdvancedPnMethod(problem, randvar_in)

    # Solve problem
    randvar_out, info = pnmethod.solve(**kwargs)

    # Return output with information (e.g. on convergence)
    return randvar_out, info
```
This interface is separate from the actual implementation(s) of the PN method. Often there are different variations of a
given numerical routine depending on the arguments supplied. These are implemented in a class hierarchy usually in the same module
as the interface. In order to decrease pesky type bugs and increase maintainability these implementations must have [type
hints](https://docs.python.org/3/library/typing.html).

```python
from typing import Tuple
from probnum import prob

class PnMethod:
    """
    Probabilistic numerical method

    Parameters
    ----------
    problem : Problem, shape=(n,n)
        Arguments defining the numerical problem to be solved.
    randvar_in : RandomVariable, shape=(n,)
        Input random variable encoding prior information about the problem.
    """

    def __init__(self, problem: Problem, randvar_in: prob.RandomVariable):
        raise NotImplementedError
    
    def solve(self, **kwargs) -> Tuple[prob.RandomVariable, dict]:
        raise NotImplementedError

class VanillaPnMethod(PnMethod):

    def __init__(self, problem: Problem, randvar_in: prob.RandomVariable):
        raise NotImplementedError

    def solve(self) -> Tuple[prob.RandomVariable, dict]:
        """
        Solve the numerical problem in a basic way.
        
        Returns
        -------
        randvar_out : RandomVariable
            Posterior distribution over the solution of `problem`.
        info : dict
            Information on the convergence of the iteration.
        """
        raise NotImplementedError


class AdvancedPnMethod(PnMethod):

    def __init__(self, problem: Problem, randvar_in: prob.RandomVariable):
        raise NotImplementedError

    def solve(self, maxiter: int) -> Tuple[prob.RandomVariable, dict]:
        """
        Solve the numerical problem in an advanced way.
        
        Parameters
        ----------
        maxiter : int
            Maximum number of iterations of the solve loop.

        Returns
        -------
        randvar_out : RandomVariable
            Posterior distribution over the solution of `problem`.
        info : dict
            Information on the convergence of the iteration.
        """
        raise NotImplementedError
```
Before you add a new method interface or class, look through the codebase whether you can simply subclass an existing 
implementation of a PN method.

## Tests


## Conclusion
Once you are done, consider writing an [example notebook](https://probabilistic-numerics.github.io/probnum/development/example_notebook.html) showcasing your new 
implementation. Congratulations you just implemented your first probabilistic numerical method in ProbNum!