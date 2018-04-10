# -*- coding: utf-8 -*-

# Metropolis Drift-Diffusion-Model
# Copyright 2018 F. Maccheroni, M. Pirazzini, C. Baldassi
# This file is released under the MIT licence that accompanies the code

## This file contains the core computational routines

import numpy as np
import ddm as dm

## Some auxiliary functions

def uniform_proposal(n, b):
    """
    choose a proposal uniformly from a set of `n` alternatives,
    excluding the incumbent `b`.
    """
    alternatives = np.arange(n)
    return np.random.choice(alternatives[np.arange(n) != b])

def nonuniform_proposal(em, b):
    """
    choose a proposal according to a symmetric stochastic matrix (the exploration matrix `em`);
    the incumbent `b` is passed as a parameter and it is used to select the correct column used for sampling
    """
    n = np.shape(em)[0]
    while True:
        a = np.random.choice(np.arange(n) , p = em[:,b])
        if a != b:
            return a

def cname(x):
    "class name, for pretty-printing purposes only"
    return x.__class__.__name__


## DDM sampling

def ddm_sample(u_a, u_b, lbarrier, ubarrier):
    """
    Sample a response time and a choice outcome in a drift-diffusion-model decision process, given
    the utilities and the decision thresholds. See also `DDMSampler` for a cached version.

    Inputs
    ------

    * `u_a`: utility of choice `a` (candidate)
    * `u_b`: utility of choice `b` (incumbent)
    * `lbarrier`: threshold for choice `a`
    * `ubarrier`: threshold for choice `b`

    Outputs
    -------

    * `RT`: a `float` representing the response time
    * `CO`: a `bool` representing the choice outcome: `True` if the proposal is accepted
      (`a` was chosen), `False` otherwise (`b` was chosen)
    """
    mu_ab = np.array([(u_a - u_b) / np.sqrt(2)])
    lbound_normarray = np.array([lbarrier / np.sqrt(2)])
    ubound_normarray = np.array([ubarrier / np.sqrt(2)])
    dt = 0.000000001 # irrelevant

    RT, CO = dm.rand_asym(mu_ab, -lbound_normarray, ubound_normarray, dt, 1)

    return RT[0], CO[0]


class DDMSampler:
    """
    A class that helps sampling response times and a choice outcomes in a
    drift-diffusion-model decision process, given a vector of utilities
    and the decision thresholds. It works essentially like a version
    of `ddm_sample` that caches the samples to improve speed (at the cost
    of memory).

    The constructor takes the following arguments:

    * `u`: a vector of utilities
    * `lb`: threshold for the incumbents
    * `ub`: threshold for the candidated
    * `cache_size`: the number of samples to cache for each combination of
      the values in `u` (default is 50)

    After creation, you can call it like a function with two arguments `a` and
    `b` (two integers, the candidate and the incumbent), and it will return
    a response time and a choice outcome (basically the same as `ddm_sample`).
    """
    def __init__(self, u, lb, ub, cache_size = 50):
        if cache_size <= 0:
            raise ValueError('invalid `cache_size`, must be positive, given %i' % cache_size)
        c = np.sqrt(2)
        n = len(u)
        self.n = n
        self.mu = [np.array([(u[a] - u[b]) / c]) for a in range(n) for b in range(n)]
        self.lb, self.ub = np.array([lb / c]), np.array([ub / c])
        self.ind = [cache_size for a in range(n) for b in range(n)]
        self.RT = [None for a in range(n) for b in range(n)]
        self.CO = [None for a in range(n) for b in range(n)]
        self.cache_size = cache_size

    def _linind(self, a, b):
        n = self.n
        return a * n + b

    def _refresh(self, a, b):
        k = self._linind(a, b)
        mu = self.mu[k]
        lb, ub, cache_size = self.lb, self.ub, self.cache_size
        dt = 0.000000001 # irrelevant
        self.RT[k], self.CO[k] = dm.rand_asym(mu, -lb, ub, dt, cache_size)
        self.ind[k] = 0

    def __call__(self, a, b):
        k = self._linind(a, b)
        if self.ind[k] == self.cache_size:
            self._refresh(a, b)
            assert self.ind[k] == 0
        RT, CO = self.RT[k][self.ind[k]], self.CO[k][self.ind[k]]
        self.ind[k] += 1
        return RT, CO


## Some auxiliary functions to check (and possiblt convert) arguments

def _check_metr_args1(u, lbarrier, ubarrier):
    if not isinstance(u, np.ndarray) or u.dtype != float:
        try:
            u = np.array(u, dtype=float)
        except:
            raise TypeError('invalid utilities list `u`, unable to convert to `float` array')
    if u.ndim != 1:
        raise ValueError('invalid utilities list `u`, should be 1-dimensional')
    if len(u) == 0:
        raise ValueError('empty utilities list')

    if not (isinstance(lbarrier, (int,float)) and isinstance(ubarrier, (int,float))):
        raise TypeError('thresholds lbarrier,ubarrier must be ints or floats, given: %s,%s' % (cname(lbarrier), cname(ubarrier)))
    lbarrier, ubarrier = float(lbarrier), float(ubarrier)
    if lbarrier <= 0 or ubarrier <= 0:
        raise ValueError('thresholds lbarrier,ubarrier must be positive, given: %f,%f' % (lbarrier, ubarrier))

    return u, lbarrier, ubarrier

def _check_metr_args2(n, t, em):
    assert isinstance(n, int) and n > 0

    if not isinstance(t, (int,float)):
        raise TypeError('time limit `t` must be an int or a float, given: %s' % cname(t))
    t = float(t)
    if t <= 0:
        raise ValueError('time limit `t` must be positive, given: %f' % t)

    if em is not None:
        if not isinstance(em, np.ndarray) or em.dtype != float:
            try:
                em = np.array(em)
            except:
                raise TypeError('invalid exploration matrix `em`, cannot convert to an array of floats')
        if em.ndim != 2:
            raise ValueError('invalid exploration matrix `em`, should be 2-dimensional')
        if em.shape != (n, n):
            raise ValueError('invalid exploration matrix `em` size, expected %i×%i, given: %i×%i' % (n, n, *em.shape))
        if np.any(np.abs(em.sum(axis=0) - 1.0) > 1e-8):
            raise ValueError('exploration matrix columns are not normalized')

    return t, em


## Metropolis-based Multiple-choice samplers

def timed_metropolis(n, sampler, t, em, *, check_args = True):
    """
    Simulate a multiple-choice decision process as a sequence of pariwise comparisons.
    At each step, a decision is made between an incumbent and a candidate; the candidates are
    proposed according to an exploration matrix; the final choice is determined
    by the current incumbent when the total available time has elapsed.

    Inputs
    ------

    * `n`: the number of possible choices.
    * `sampler`: this must be a function (or an object) that takes two integer arguments (an
      incumbent and a candidate, both between `0` and `n-1`) and returns two items: a response
      time (a `float`) and a `bool` determining whether the proposal was accepted.
    * `t`: time limit
    * `em`: exploration matrix. If `None`, candidates are extracted uniformly at random.
      Otherwise, it should be a `n`×`n` matrix.
    * `check_args`: (keyword-only argument) whether to check/convert the arguments, defaults to
      `True`.

    Output
    ------

    The output is the index corresponding to the final choice, an `int` between `0` and `n-1`.
    """
    if check_args:
        t, em =  _check_metr_args2(n, t, em)

    if em is None:
        proposal = lambda b: uniform_proposal(n, b)
    else:
        proposal = lambda b: nonuniform_proposal(em, b)

    s = 0.0                       # clock
    b = np.random.randint(n)      # initial choice (set the incumbent)
    while True:
        a = proposal(b)           # candidate
        RT, CO = sampler(a, b)    # decision
        s += RT                   # increase clock
        if s > t:
            break                 # time is over
        elif CO:
            b = a                 # accept the proposal (update incumbent)
    return b


def metropolis_ddm_hist(u, lbarrier, ubarrier, t, em = None, num_samples = 10**3, cache_size = 50):
    """
    Call `timed_metropolis` repeatedly, using a drift-diffusion model as the pairwise sampler,
    and return a count of the occurrences of each outcome.

    Inputs
    ------

    * `u`: a vector of utilities (anything that can be converted to a numpy 1-d array of
      floats is acceptable
    * `lbarrier`: threshold for the incumbents
    * `ubarrier`: threshold for the candidates
    * `t`: time limit
    * `em`: exploration matrix. See the docs for `timed_metropolis`. Defaults to `None`.
    * `num_samples`: number of tests to perform. Defaults to `10**3`.
    * `cache_size`: if positive, it's the number of samples that get cached in advance for each
      possible combination of the pairwise choices (improves performance, but requires O(n^2) memory);
      if zero, disables caching. Defaults to `50`.

    Output
    ------

    A vector (a 1-d numpy array of integers) with the same length as the input `u`,
    in which each entry counts the number of times an item was chosen (out of the
    `num_samples` tests).
    """

    u, lbarrier, ubarrier = _check_metr_args1(u, lbarrier, ubarrier)
    n = len(u)
    t, em = _check_metr_args2(n, t, em)

    if not isinstance(num_samples, int):
        raise TypeError('number of samples `num_samples` must be an int, given: %s' % cname(num_samples))
    if num_samples < 0:
        raise ValueError('number of samples `num_samples` must be non-negative, given: %i' % num_samples)

    if not isinstance(cache_size, int):
        raise TypeError('invalid `cache_size`, expected an `int`, given: %s' % cname(cache_size))
    if cache_size < 0:
        raise ValueError('the `cache_size` must be non-negative, given: %i' % cache_size)

    if cache_size > 0:
        ddm_sampler = DDMSampler(u, lbarrier, ubarrier, cache_size)
    else:
        ddm_sampler = lambda a, b: ddm_sample(u[a], u[b], lbarrier, ubarrier)

    choice_count = np.zeros(n, dtype=int)
    for samples in range(num_samples):
        choice = timed_metropolis(n, ddm_sampler, t, em, check_args=False)
        choice_count[choice] += 1
    return choice_count
