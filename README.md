# Uncnstrained-Aircraft-Structure-Optimisation
Basic algorithms for unconstrained aircraft structure optmisation

Implement the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) and its analytic gradient in `aso.problem_factory.ProblemFactory.rosenbrock`.

Then, implement a convergence check in `aso.optimiser.Optimiser.converged` and a steepest descent algorithm in `aso.optimiser.Optimiser.steepest_descent`.

Ensure that your virtual environment is activated and run
```
pytest -m project_1
```

This will apply your algorithm to seven test problems. A test is passed if your steepest descent implementation converges to a minimum in less than 100,000 iterations and failed otherwise.

#### Note on Implementation

The tests expect the `Optimiser` to modify the design variables in place. Hence, you have to update them using one of the following options.

In place, efficient, intuitive:
```
self.x += step_size * search_direction
```

Equivalent but less intuitive:
```
numpy.add(self.x, step_size * search_direction, out=self.x)
```

Less efficient because it creates a new, temporary array for the right-hand side, copies it back to the left-hand side, and discards the temporary right-hand side:
```
self.x[:] = self.x + step_size * search_direction
```

The following option, on the other hand, would not pass the tests, since `x` will be pointing to a different location in memory after each iteration:
```
self.x = self.x + step_size * search_direction
```

#### Note on Debugging

By default, `pytest` intercepts `sys.stdout`, so you cannot `print` to the console during tests. If, however, you want to use `print` statements for debugging in combination with `pytest`, just add the `-s` flag to the `pytest` command. It's a shortcut for `--capture=no`, which disables all [capturing](https://docs.pytest.org/en/latest/how-to/capture-stdout-stderr.html), so your `print` statements will be visible on the console again.

Alternatively, you can write log messages, which will be saved to the logs directory. For example,
```
logger.debug(f"iteration = {iteration}, x = {self.x}")
```

You can also use the debugging features of your IDE. There is a Python Debugger extension by Microsoft for VS Code, for example.

#### Hint

If your algorithm does not pass some or even all of the tests, the problem could be the algorithm itself, but it could also be a parameter such as the step size, especially if it is constant. Try different values to find one that is suitable for all seven test problems.


