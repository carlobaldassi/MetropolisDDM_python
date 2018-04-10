# -*- coding: utf-8 -*-

# Metropolis Drift-Diffusion-Model
# Copyright 2018 F. Maccheroni, M. Pirazzini, C. Baldassi
# This file is released under the MIT licence that accompanies the code

## This file contains a console interface and related routines

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.csgraph import connected_components

import metropolis_ddm

class Abort(Exception):
    pass

def isintopt(x):
    return x is None or isinstance(x, int)

def isstropt(x):
    return x is None or isinstance(x, str)

def input_int(msg, *, lb = None, ub = None, default = None):
    assert isintopt(lb) and isintopt(ub) and isintopt(default)
    if default is not None:
        assert lb is None or default >= lb
        assert ub is None or default <= ub
        msg += ' [default=%i]' % default
    msg += ': '

    while True:
        x = input(msg).strip()
        if default is not None and len(x) == 0:
            return default
        try:
            v = int(x)
        except ValueError:
            print('Invalid input, please enter an integer')
            continue
        if lb is not None and v < lb:
            print('Invalid value, expected v >= %i, given: %i' % (lb, v))
            continue
        if ub is not None and v > ub:
            print('Invalid value, expected v <= %i, given: %i' % (ub, v))
            continue
        break
    return v

def input_bool(q, *, default=True, yes=['y', 'yes', '1'], no=['n', 'no', '0']):
    yes = yes if isinstance(yes, list) else [yes]
    no = no if isinstance(no, list) else [no]
    y0, n0 = yes[0], no[0]
    deft, deff = -1, -1
    if default is True:
        opts = '[%s]/%s' % (y0, n0)
        deft = 0
    elif default is False:
        opts = '%s/[%s]' % (y0, n0)
        deff = 0
    else:
        opts = '%s/%s' % (y0, n0)
    msg = q + (' (%s)? ' % opts)
    while True:
        x = input(msg).upper().strip()
        if len(x) == deft or x in map(lambda s: s.upper(), yes):
            return True
        if len(x) == deff or x in map(lambda s: s.upper(), no):
            return False
        print("Invalid input, please enter '%s' or '%s'" % (y0, n0))

def isfloatopt(x):
    return x is None or isinstance(x, float)

def input_float(msg, *, lbt = None, lbe = None, ubt = None, ube = None, default = None):
    assert lbt is None or lbe is None
    assert ubt is None or ube is None
    assert isfloatopt(lbt) and isfloatopt(lbe) and isfloatopt(ubt) and isfloatopt(ube) and isfloatopt(default)
    if default is not None:
        assert lbe is None or default >= lbe
        assert lbt is None or default > lbt
        assert ube is None or default <= ube
        assert ubt is None or default < ubt
        msg += ' [default=%g]' % default
    msg += ': '

    while True:
        x = input(msg).strip()
        if default is not None and len(x) == 0:
            return default
        try:
            v = float(x)
        except ValueError:
            print('Invalid input, please enter a float')
            continue
        if v != v:
            print('Invalid input, `nan` is not allowed')
            continue
        if lbt is not None and v <= lbt:
            print('Invalid value, expected v > %g, given: %g' % (lbt, v))
            continue
        if lbe is not None and v < lbe:
            print('Invalid value, expected v >= %g, given: %g' % (lbe, v))
            continue
        if ubt is not None and v >= ubt:
            print('Invalid value, expected v < %g, given: %g' % (ubt, v))
            continue
        if ube is not None and v > ube:
            print('Invalid value, expected v <= %g, given: %g' % (ube, v))
            continue
        break
    return v

def input_indices(msg, n, m = None, *, filt = None, abort = None):
    assert isinstance(msg, str) and isinstance(n, int) and isintopt(m) and isstropt(abort)
    if m is None:
        m = n
    if abort is not None:
        msg += " ['%s' to abort]" % abort
    msg += ': '
    while True:
        s = input(msg).strip()
        if abort is not None and s == abort:
            raise Abort()
        ls = s.split(',')
        if len(ls) != 2:
            print('Invalid format, you need to enter two indices separated by a comma')
            continue
        try:
            i, j = int(ls[0]), int(ls[1])
        except ValueError:
            print('Invalid indices, please enter two integers')
            continue
        if not (0 <= i < n and 0 <= j < m):
            print('Out of bounds, matrix size is %s×%s, given: %i,%i' % (i, j))
            continue
        if filt is not None and not filt(i, j):
            print('Invalid choice %i,%i' % (i, j))
        break
    return i, j


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

    mask = exp_matr != dist # boolean mask (matrix)

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

def explo_matrix_input(n, ro, alt):
    if not isinstance(n, int):
        raise TypeError('argument `n` must be an int, given: %s' % cname(n))
    if n <= 0:
        raise ValueError('argument `n` must be > 0, given: %i' % n)
    if not isinstance(ro, float):
        raise TypeError('argument `ro` must be a float, given: %s' % cname(ro))
    if ro < 0:
        raise ValueError('argument `ro` must be >= 0, given: %g' % ro)
    if alt not in (0,1):
        raise ValueError('argument `alt` must be 0 or 1, given: %s' % alt)

    dist = np.ones((n,n))
    dist[np.diag_indices(n)] = 0
    mask = dist != 0

    if alt: # DISTANCE
        print('Input the distance matrix of the alternatives:')

        g_aux = dist.copy() #corresponding graph of the distance matrix, will be used to check connected components

        while True:
            print('Current Distance Matrix:\n%s' % dist)

            try:
                a, b = input_indices('Choose two indices (i,j)', n, filt = lambda i,j: i != j, abort = 'q')
            except Abort:
                break

            d = input_float('Enter new d[%i,%i]' % (a, b), lbt = 0.0)
            if np.isposinf(d):
                g_aux[a,b] = 0
                g_aux[b,a] = 0

            if connected_components(g_aux)[0] != 1:
                print('WARNING! This action disconnects the set of alternatives.')
                print('This is not allowed. The previous distance is maintained')
                g_aux[a,b] = 1
                g_aux[b,a] = 1
            else:
                dist[a,b] = d
                dist[b,a] = d

        dist /= np.min(dist[mask]) #normalization

    else: #GRAPH
        print('Input the graph of the alternatives:')

        graph = dist.astype(int)  #makes a copy of dist consisting of integers

        while True:
            print('Current Graph:\n%s' % graph)

            try:
                a, b = input_indices('Choose two indices (i,j) to flip', n, filt = lambda i,j: i != j, abort = 'q')
            except Abort:
                break

            d = 1 - graph[a,b]
            graph[a,b] = d
            graph[b,a] = d

            if d == 0 and connected_components(graph)[0] != 1:
                print('WARNING! This action disconnects the set of alternatives.')
                print('This is not allowed. The previous situation is maintained')
                graph[a,b] = 1
                graph[b,a] = 1

        print('Final Graph:')
        print(graph)
        dist = floyd_warshall(graph)
        print()

    print('Final Distance Matrix:')
    print(dist)

    exp_matr = np.zeros((n,n))
    exp_matr[mask] = (1/(n-1)) * 1/(dist[mask]**ro)
    exp_matr[~mask] = 1 - (np.sum(exp_matr, axis=0) - np.diag(exp_matr))

    return np.abs(exp_matr)


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
    n = input_int('Number of alternatives', lb = 1, default = 5)

    u = np.arange(n, dtype=float)

    if input_bool('Advanced utility options', default = False):
        print('Input utilities: ')
        u = [input_float('u[%i]' % i) for i in range(n)]
        u = np.array(u)

    u *= 7.071 / (np.max(u) - np.min(u))

    t = input_float('Time limit', lbt = 1.0, ubt = np.inf, default = 2.0)

    upper_barrier = np.log(t+1) / (np.max(u) - np.min(u))
    lower_barrier = upper_barrier

    if input_bool('Advanced threshold settings', default = False):
        upper_barrier = input_float('Input acceptance threshold', lbt = 0.0)
        lower_barrier = input_float('Input rejection threshold', lbt = 0.0)

    if input_bool('Advanced exploration settings', default = False):
        ro = input_float('Exploration aversion parameter', lbt = 0.0, default = 1.0)
        alt = input_bool('Distance or Graph', yes=['d', '1'], no=['g', '0'])
        em = explo_matrix_input(n, ro, alt)
    else:
        em = None

    num_samples = input_int('Number of samples', lb = 0, default = 10**3)

    print()
    print(n, 'alternatives with normalized utilities', u, 'choose in', t, 'time units\n')
    print('Acceptance/rejection thresholds: [%g, %g]\n' % (upper_barrier, lower_barrier))

    if em is not None:
        print('Exploration matrix:\n%s\n' % em)
    else:
        print('Exploration matrix:\n%s\n' % explo_matrix_unif(n))

    p = np.exp(upper_barrier*u) / np.sum(np.exp(upper_barrier*u)) # softmax

    choice_count = metropolis_ddm.metropolis_ddm_hist(u, lower_barrier, upper_barrier, t, em, num_samples)

    choice_freq = choice_count / np.sum(choice_count)

    print('Choice count: %s\n' % choice_count)
    print('Total variation distance: %g\n' % (np.linalg.norm(choice_freq - p, 1) / 2))
    print('Maximum simulation error: %g' % np.max(np.abs(choice_freq - p)))

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
