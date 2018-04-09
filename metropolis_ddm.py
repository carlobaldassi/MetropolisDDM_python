# Metropolis Drift-Diffusion-Model
# Copyright 2018 F. Maccheroni, M. Pirazzini, C. Baldassi
# This file is released under the MIT licence that accompanies the code

import numpy as np
import matplotlib.pyplot as plt
import ddm as dm

from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.csgraph import connected_components

# function that chooses a proposal uniformly from the set of alternatives (n)
# that are different from the incumbent (b)

def uniform_proposal(n, b):
    alternatives = np.arange(n)
    return np.random.choice(alternatives[np.arange(n) != b])

# function that chooses a proposal according to a symmetric stochastic matrix (the exploration matrix 'em')
# the incumbent is passed as a parameter (b) and it is used to select the correct column used for sampling

def nonuniform_proposal(em, b):
    n = np.shape(em)[0]
    while True:
        a = np.random.choice(np.arange(n) , p = em[:,b])
        if a != b:
            return a

def cname(x):
    return x.__class__.__name__

# UNIFORM EXPLORATION MATRIX
# this function is not actually used for computation (we use uniform_proposal in that case),
# it is used for plotting the exploration matrix when it is uniform

def explo_matrix_unif(n):
    if not isinstance(n, int):
        raise TypeError('argument `n` must be an int, given: %s' % cname(n))
    if n < 2:
        raise ValueError('argument `n` must be >= 2, given: %i' % n)

    dist = np.ones((n,n))
    dist[np.diag_indices(n)] = 0

    exp_matr = np.zeros((n,n))

    mask = exp_matr != dist #boolean mask (matrix)

    exp_matr[mask] = (1 / (n-1)) / dist[mask]
    exp_matr[~mask] = 1 - (np.sum(exp_matr, axis=0) - np.diag(exp_matr))
                         # np.sum(exp_matr,axis=0) -> array of the sums of the columns of exp_matr

    return exp_matr

# EXPLORATION MATRIX INPUT
# This function lets the user input the exploration matrix, either through a distance
# matrix or through a graph. When the user decides to input a graph we later convert
# it to a distance matrix and then transorm it into an exploration matrix.
# The arguments of the function are the number of alternatives (n), the exploration
# aversion parameter (ro) and a parameter that tells the function whether to work
# on a graph or directly on a distance matrix (alt, 1 for distance and 0 for graph)

# The input procedure works as follows:
# 0 - both the graph and the distance matrix are initialized as uniform, so as if the
#     corresponding graph were complete with weights equal to 1
# 1 - choose a pair of nodes (alternatives), they must be written in the form (i,j)
# 2 - then, once the function checked whether the proposed alternatives are valid,
#     the user can insert the value to put in chosen position. If the user is working
#     on a graph the value can be either 0 or 1, otherwise it can be any positive float
#     (for the distance matrix), but it cannot disconnect the final graph
# 3 - To stop the construction of the matrix, the user simply has to input '0' whenever
#     'Continue (0/1)?' is asked.
#
# After the input procedure, the distance matrix is normalized so that its minimum entry
# outside the main diagonal is 1 (we do not need to do so for the graph because by construction
# the minimum distance is already 1).
#

def explo_matrix_input(n, ro, alt):
    if not isinstance(n, int):
        raise TypeError('argument `n` must be an int, given: %s' % cname(n))
    if n <= 0:
        raise ValueError('argument `n` must be > 0, given: %i' % n)
    if not isinstance(ro, float):
        raise TypeError('argument `ro` must be a float, given: %s' % cname(ro))
    if ro < 0:
        raise ValueError('argument `ro` must be >= 0, given: %f' % ro)
    if alt not in (0,1):
        raise ValueError('argument `alt` must be 0 or 1, given: %s' % alt)

    dist = np.ones((n,n))
    dist[np.diag_indices(n)] = 0
    mask = dist != 0

    if alt:                                                     # DISTANCE

        g_aux = dist.copy() #corresponding graph of the distance matrix, will be used to check connected components

        while True:
            print('Current Distance Matrix:\n%s' % dist)
            s = input('Alternatives (a,b): ')

            while True:
                ls = s.split(',')
                if len(ls) == 2 and ls[0].isdigit() and ls[1].isdigit():
                    break
                else:
                    print('Invalid alternatives input form. It must be of the type (a,b) with a and b integers')
                    s = input('Alternatives (a,b): ')

            a,b = int(ls[0]),int(ls[1])

            if not (0 <= a < n): # if a >= n or a < 0:
                print('Invalid alternative, a must be an integer between 0 and %i' % (n-1))
                c = int(input('Continue matrix adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break

            if not (0 <= b < n):
                print('Invalid alternative, b must be an integer between 0 and %i' % (n-1))
                c = int(input('Continue matrix adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break

            if a == b:
                print('The distance between an alternative and itself cannot be modified (0 by definition of semi-metric).')
                c = int(input('Continue matrix adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break

            if dist[a,b] != 1:
                o = int(input('Distance already adjusted, overwrite (0/1)? '))
                if not o:
                    continue

            while True:
                d = input('New distance between %i and %i? ' % (a, b))
                if d in ('inf', 'np.inf'):
                    d = np.inf
                    g_aux[a,b] = 0
                    g_aux[b,a] = 0
                    break
                elif float(d) <= 0:
                    print('Invalid distance, it must be a positive float')
                else:
                    d = float(d)
                    break

            cc1 = connected_components(g_aux)
            if cc1[0] != 1:
                print('WARNING! This action disconnects the set of alternatives.')
                print('This is not allowed. The previous distance is maintained')
                g_aux[a,b] = 1
                g_aux[b,a] = 1
            else:
                dist[a,b] = d
                dist[b,a] = d

            c = int(input('Continue matrix adjustments (0/1)? '))
            if c:
                continue
            else:
                break

        dist /= np.min(dist[mask]) #normalization


    else:                                                           #GRAPH
        graph = dist.astype(int)  #makes a copy of dist consisting of integers

        while True:
            print('Current Graph:\n%s' % graph)

            s = input('Alternatives (a,b): ')

            while True:
                ls = s.split(',')
                if len(ls) == 2 and ls[0].isdigit() and ls[1].isdigit():
                    break
                else:
                    print('Invalid alternatives input form. It must be of the type (a,b) with a and b integers')
                    s = input('Alternatives (a,b): ')

            a,b = int(ls[0]),int(ls[1])

            if not (0 <= a < n):
                print('Invalid alternative, a must be an integer between 0 and %i' % (n-1))
                c = int(input('Continue graph adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break

            if not (0 <= b < n):
                print('Invalid alternative, b must be an integer between 0 and %i' % (n-1))
                c = int(input('Continue graph adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break

            if a == b:
                print('Invalid alternatives, no edges from a vertex to itself are allowed.')
                c = int(input('Continue graph adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break

            while True:
                d = int(input('Connect %i and %i (0/1)? ' % (a, b)))
                if d not in (0,1):
                    print('Invalid value, it must be either 0 or 1')
                else:
                    break

            graph[a,b] = d
            graph[b,a] = d

            cc1 = connected_components(graph)
            if cc1[0] != 1:
                print('WARNING! This action disconnects the set of alternatives.')
                print('This is not allowed. The previous situation is maintained')
                graph[a,b] = 1
                graph[b,a] = 1

            c = int(input('Continue graph adjustments (0/1)? '))
            if c:
                continue
            else:
                break

        print('Final Graph:')
        print(graph)
        #FLOYD WARSHALL
        dist = floyd_warshall(graph)

    print('Final Distance Matrix:')
    print(dist)

    exp_matr = np.zeros((n,n))
    exp_matr[mask] = (1/(n-1)) * 1/(dist[mask]**ro)
    exp_matr[~mask] = 1 - (np.sum(exp_matr,axis=0) - np.diag(exp_matr))

    return np.abs(exp_matr)


## DDM sampling

class DDMSampler:
    """
    A class that helps sampling response times and a choice outcomes in a
    drift-diffusion-model decision process, given a vector of utilities
    and the decision thresholds. It works essentially like a cached version
    of `ddm_sample`.

    The constructor takes the following arguments:

    * `u`: a vector of utilities
    * `lb`: threshold for the incumbents
    * `ub`: threshold for the candidated
    * `cache_size`: the number of samples to cache for each combination of
      the values in `u`.

    It provides one relevant method, `sample(a, b)`.
    """
    def __init__(self, u, lb, ub, cache_size = 50):
        if cache_size <= 0:
            raise ValueError('invalid `cache_size`, must be positive, given %i' % cache_size)
        c = np.sqrt(2)
        n = len(u)
        self.mu = [np.array([(u[a] - u[b]) / c]) for a in range(n) for b in range(n)]
        self.n = n
        self.lb, self.ub = np.array([lb / c]), np.array([ub / c])
        self.ind = [cache_size for a in range(n) for b in range(n)]
        self.cache_size = cache_size

    def refresh(self, a, b):
        n = self.n
        k = a * n + b
        mu = self.mu[k]
        lb, ub, cache_size = self.lb, self.ub, self.cache_size
        dt = 0.000000001 # irrelevant
        self.RT, self.CO = dm.rand_asym(mu, -lb, ub, dt, cache_size)
        self.ind[k] = 0

    def sample(self, a, b):
        """
        Returns a sample (a response time and a choice outcome) for two items `a`
        and `b` (two integers, the incumbent and the candidate).
        Basically the same as `ddm_sample`.
        """
        n = self.n
        k = a * n + b
        if self.ind[k] == self.cache_size:
            self.refresh(a, b)
            assert self.ind[k] == 0
        RT, CO = self.RT[self.ind[k]], self.CO[self.ind[k]]
        self.ind[k] += 1
        return RT, CO

def ddm_sample(u_a, u_b, lbarrier, ubarrier):
    """
    Sample a response time and a choice outcome in a drift-diffusion-model decision process, given
    the utilities and the decision thresholds. See also `DDMSampler` for a cached version.

    Inputs
    ------

    * `u_a`: utility of choice `a` (incumbent)
    * `u_b`: utility of choice `b` (candidate)
    * `lbarrier`: threshold for choice `a`
    * `ubarrier`: threshold for choice `b`

    Outputs
    -------

    * `RT`: a `float` representing the response time
    * `CO`: a `bool` representing the choice outcome: `True` if the proposal is accepted
      (`b` was chosen), `False` otherwise (`a` was chosen)
    """
    mu_ab = np.array([(u_a - u_b) / np.sqrt(2)])
    lbound_normarray = np.array([lbarrier / np.sqrt(2)])
    ubound_normarray = np.array([ubarrier / np.sqrt(2)])
    dt = 0.000000001 # irrelevant

    RT, CO = dm.rand_asym(mu_ab, -lbound_normarray, ubound_normarray, dt, 1)

    return RT[0], CO[0]

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

    s = 0.0                  # clock
    b = np.random.randint(n) # initial choice
    while True:
        a = proposal(b)
        RT, CO = sampler(a, b)
        s += RT
        if s > t:
            break
        elif CO:
            b = a
    return b


def metropolis_ddm_hist(u, lbarrier, ubarrier, t, em = None, num_samples = 10**3, cached_sampler = True):
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
    * `cached_sampler`: whether to chache the samples (improves performance, but requires O(n^2) memory).

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

    if not isinstance(cached_sampler, bool):
        raise TypeError('invalid argument `cached_sampler`, expected a `bool`, given: %s' % cname(cached_sampler))

    if cached_sampler:
        ddm = DDMSampler(u, lbarrier, ubarrier)
        ddm_sampler = lambda a, b: ddm.sample(a, b)
    else:
        ddm_sampler = lambda a, b: ddm_sample(u[a], u[b], lbarrier, ubarrier)

    choice_count = np.zeros(len(u), dtype=int)
    for samples in range(num_samples):
        choice = timed_metropolis(n, ddm_sampler, t, em, check_args=False)
        choice_count[choice] += 1
    return choice_count

def run_comparison():
    """
    An interactive function to run a comparison between a Metropolis-DDM multiple-choice
    simulation and the limiting distribution given by the softmax of the utilities.

    All parameters and settings are input interactively. At the end of the simulation,
    a plot comparing the frequencies is produced, and some summarizing information is printed
    on the console.
    """
    # This code contains several loops to ensure that the user inputs the right type
    # of parameters, however some checks are omitted for simplicity.
    while True:
        n = int(input('Number of alternatives: '))
        if n <= 0:
            print('The number of alternatives must be a positive integer')
        else:
            print('Your alternatives are 0,1,...,%i' % (n-1))
            break

    u = np.arange(n, dtype=float)

    while True:
        u_options = input('Advanced utility options (0/1)? ')
        if int(u_options) in (0, 1):
            break
        else:
            print('Please enter a valid input (0/1)')

    if int(u_options) == 1:
        print('Input utilities: ')
        u = [float(input('u(%i) = ' % i)) for i in range(n)]
        u = np.array(u)

    u *= 7.071 / (np.max(u) - np.min(u))

    while True:
        t = float(input('Time limit: '))
        if t > 1:
            break
        else:
            print('The time limit needs to be strictly greater than 1')

    upper_barrier = np.log(t+1) / (np.max(u) - np.min(u))

    lower_barrier = upper_barrier

    while True:
        barrier_options = input('Advanced threshold settings (0/1)? ')
        if int(barrier_options) in (0, 1):
            break
        else:
            print('Please enter a valid input (0/1)')

    if int(barrier_options) == 1:
        upper_barrier = float(input('Input acceptance threshold: '))
        lower_barrier = float(input('Input rejection threshold: '))

    while True:
        q_o = input('Advanced exploration settings (0/1)? ')
        if int(q_o) in (0, 1):
            q_o = int(q_o)
            break
        else:
            print('Please enter a valid input (0/1)')

    if q_o:
        ro = float(input('Exploration aversion parameter (input a positive real number): '))
        alt = int(input('Graph (0) or Distance (1)? '))
        if alt:
            print('Input the distance matrix of the alternatives:')
        else:
            print('Input the graph of the alternatives:')
        em = explo_matrix_input(n,ro,alt)
    else:
        em = None

    while True:
        num_samples = int(input('Number of samples: '))
        if num_samples <= 0:
            print('The number of samples should be a positive integer')
        else:
            break

    print()
    print(n, 'alternatives with normalized utilities', u, 'choose in', t, 'time units\n')
    print('Acceptance/rejection thresholds: [%f, %f]\n' % (upper_barrier, lower_barrier))

    if q_o:
        print('Exploration matrix:\n%s\n' % em)
    else:
        print('Exploration matrix:\n%s\n' % explo_matrix_unif(n))

    p = np.exp(upper_barrier*u) / np.sum(np.exp(upper_barrier*u)) # softmax

    choice_count = metropolis_ddm_hist(u, lower_barrier, upper_barrier, t, em, num_samples)

    choice_freq = choice_count / np.sum(choice_count)

    print('Choice count: %s\n' % choice_count)
    print('Total variation distance: %f\n' % (np.linalg.norm(choice_freq - p, 1) / 2))
    print('Maximum simulation error: %f' % np.max(np.abs(choice_freq - p)))

    # COMPARISON PLOT

    fig = plt.figure()
    ax = fig.add_subplot(111)
    a = np.arange(n)
    # labels = [str(u[i]) for i in range(n)]
    ax.set_xticks(a)
    # ax.set_xticklabels(labels)

    plt.plot(a, p, label = 'softmax')  # plot softmax888

    plt.plot(a, choice_freq, label = 'simulation') #final choice frequencies LLN for IID copies of algorithm
    plt.legend()
    plt.show()
