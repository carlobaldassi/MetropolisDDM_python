# Metropolis Drift Diffusion Model

This repository contains code that can simulate the Metropolis-Drift-Diffusion Model.

### Installation

This code is written for Python 3.

It requires the [ddm module](https://github.com/DrugowitschLab/dm), which should be installed manually prior to downloading this code.

After that, simply download this code in a local directory and `import` its modules from within that
directory (see below).

### Usage

The code is throughly commented and the documentation is accessible via docstrings (see below).

To perform an interactive simulation, import the `interface_console` module and run the
`run_comparison()` function.  You will be asked a series of questions on the parameters to use,
then a simulation will be run and the results plotted.

The core algorithms are in the `metropolis_ddm` module. The `timed_metropolis` function performs
one simulation run.  The `metropolis_ddm_hist` function runs the model repeatedly and returns a
count of the number of times each alternative was chosen.

For example to run <i>10³</i> simulations with 4 alternatives, utilities <i>(1, 3, 5, 7)</i>, lower
(incumbent) threshold at <i>-1</i>, upper (candidate) threshold at <i>2</i>, time limit <i>5</i>,
with uniform exploration matrix (the default), you can use:

```python
In [1]: from metropolis_ddm import metropolis_ddm_hist

In [2]: metropolis_ddm_hist([1.0, 3.0, 5.0, 7.0], 1.0, 2.0, 5.0, num_samples=10**3)
Out[2]: array([  0,   4,  48, 948])
```

In this batch of simulations, alternative 1 was never chosen, alternative 2 was chosen 0.4% of the
times, etc.

For more help, try:

```
In [1]: import metropolis_ddm as mddm

In [2]: mddm.DDMSampler?
In [2]: mddm.timed_metropolis?
In [3]: mddm.metropolis_ddm_hist?
