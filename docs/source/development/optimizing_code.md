# Optimizing Code

When implementing new probabilistic numerical methods consider the tips below to ensure efficiency of your code.

## Writing Efficient Python Code

1. **Make it work**: write a first implementation that does what it is supposed to do.
2. **Make it reliable**: test your code with unit tests. Ensure that any corner cases are covered by your implementation and that if you break your algorithm, that the tests capture this.
3. **Optimize the code by profiling**: find the bottlenecks in your implementation using a code profiler. Keep in mind that there is a trade off between profiling on a realistic example and the simplicity and execution speed of the code.
4. **Replace expensive functions with compiled code**: use Cython to replace computationally costly parts, such as loops.

### Using Cython

[Cython](https://cython.org/) allows the user to write code very similar to native Python, which is then compiled to C. This can reduce the amount of computing resources used for a specific function.