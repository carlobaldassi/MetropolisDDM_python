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

def uniform_proposal(n,b):
    alternatives = np.arange(n)
    return np.random.choice(alternatives[np.arange(n) != b])

# function that chooses a proposal according to a symmetric stochastic matrix (the exploration matrix 'em')
# the incumbent is passed as a parameter (b) and it is used to select the correct column used for sampling

def nonuniform_proposal(em,b):
    N = np.shape(em)[0]
    return np.random.choice(np.arange(N) , p = em[:,b])

# UNIFORM EXPLORATION MATRIX
# this function is not actually used for computation (we use uniform_proposal in that case),
# it is used for plotting the exploration matrix when it is uniform 
  
def explo_matrix_unif(n):
    if not(isinstance(n,int) and n > 0):
        raise Exception('The argument of the exploration matrix must be a positive integer!')
    
    dist = np.ones((n,n))
    dist[np.diag_indices(n)] = 0
    
    exp_matr = np.zeros((n,n))
    
    mask = exp_matr != dist #boolean mask (matrix)
    
    exp_matr[mask] = (1/(n-1)) * 1/(dist[mask])
    exp_matr[~mask] = 1 - ( np.sum(exp_matr,axis=0) - np.diag(exp_matr) ) 
                            # np.sum(exp_matr,axis=0) array of the sums of the columns of exp_matr
    
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

def explo_matrix_input(n,ro,alt):
    if not(isinstance(n,int) and n > 0):
        raise Exception('The size of the exploration matrix must be a positive integer!')
    if not(isinstance(ro,float) and ro >= 0):
        raise Exception('The exploration aversion parameter must be a positive float!')
    if alt not in (0,1):
        raise Exception('Invalid type parameter!')
    
    dist = np.ones((n,n))
    dist[np.diag_indices(n)] = 0
    mask = dist != 0
    
    if alt:                                                     # DISTANCE   
        
        g_aux = dist.copy() #corresponding graph of the distance matrix, will be used to check connected components
        
        while True:
            print('Current Distance Matrix:')
            print(dist)
            s = input('Alternatives (a,b): ')
            
            while True:
                ls = s.split(',')
                if len(ls) == 2 and ls[0].isdigit() and ls[1].isdigit():
                    break
                else:
                    print('Invalid alternatives input form! It must be of the type (a,b) with a and b integers')
                    s = input('Alternatives (a,b): ')
                              
            a,b = int(ls[0]),int(ls[1])
            
            if not(0 <= a < n): # if a >= n or a < 0:
                print('Invalid alternative, a must be an integer between 0 and', n-1,'!')
                c = int(input('Continue matrix adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break
                
            if not(0 <= b < n):
                print('Invalid alternative, b must be an integer between 0 and', n-1,'!')
                c = int(input('Continue matrix adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break
                
            if a == b:
                print('The distance between an alternative and itself cannot be modified (0 by definition of semi-metric)!')
                c = int(input('Continue matrix adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break
                
            if dist[a,b] != 1:
                o = int(input('Distance already adjusted, overwrite (0/1)? '))
                if not(o):
                    continue
                
            while True:
                d = input('New distance between ' + str(a) + ' and ' + str(b) + '? ')
                if d in ('inf','np.inf'):
                    d = np.inf
                    g_aux[a,b] = 0
                    g_aux[b,a] = 0
                    break
                elif float(d) <= 0:
                    print('Invalid distance! It must be a positive float!')
                else:
                    d = float(d)
                    break
            
            cc1 = connected_components(g_aux)
            if cc1[0] != 1:
                print('WARNING! This action disconnects the set of alternatives!')
                print('This is not allowed! The previous distance is maintained')
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
            
        dist = dist/np.min(dist[mask]) #normalization 
        
        
    else:                                                           #GRAPH
        graph = dist.astype(int)  #makes a copy of dist consisting of integers
        
        while True:
            print('Current Graph:')
            print(graph)
            
            s = input('Alternatives (a,b): ')
            
            while True:
                ls = s.split(',')
                if len(ls) == 2 and ls[0].isdigit() and ls[1].isdigit():
                    break
                else:
                    print('Invalid alternatives input form! It must be of the type (a,b) with a and b integers')
                    s = input('Alternatives (a,b): ')
                              
            a,b = int(ls[0]),int(ls[1])
                       
            if not(0 <= a < n):
                print('Invalid alternative, a must be an integer between 0 and', n-1,'!')
                c = int(input('Continue graph adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break
                
            if not(0 <= b < n):
                print('Invalid alternative, b must be an integer between 0 and', n-1,'!')
                c = int(input('Continue graph adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break
            
            if a == b:
                print('Invalid alternatives, no edges from a vertex to itself are allowed!')
                c = int(input('Continue graph adjustments (0/1)? '))
                if c:
                    continue
                else:
                    break
            
            while True:
                d = int(input('Connect ' + str(a) + ' and ' + str(b) + ' (0/1)? '))
                if d not in (0,1):
                    print('Invalid value! It must be either 0 or 1')
                else:
                    break
                
            graph[a,b] = d
            graph[b,a] = d
            
            cc1 = connected_components(graph)
            if cc1[0] != 1:
                print('WARNING! This action disconnects the set of alternatives!')
                print('This is not allowed! The previous situation is maintained')
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
    exp_matr[~mask] = 1 - ( np.sum(exp_matr,axis=0) - np.diag(exp_matr) ) 
        
    return np.abs(exp_matr)
        
        
## DDM comparison

def DDMcomparison(u_a, u_b, lbarrier, ubarrier):
    mu_ab = np.array([float( (u_a-u_b)  / np.sqrt(2) ) ])
    lbound_normarray = np.array([ lbarrier / np.sqrt(2) ])
    ubound_normarray = np.array([ ubarrier / np.sqrt(2) ])
    dt = 0.000000001

    RT, CO = dm.rand_asym(mu_ab, -lbound_normarray, ubound_normarray, dt, 1)
    
    return RT[0], CO[0]

# Given proposal a with utility u(a) = u_a, incumbent b with utility u(b) = u_b
# lower threshold \beta = lbarrier > 0 and upper threshold \lambda = ubarrier > 0 
# DDMcomparison(u_a, u_b, lbarrier, ubarrier) returns the response time RT
# and choice outcome CO = True if proposal a is accepted CO = False otherwise 
# The code contains several loops to ensure that the user inputs the right type 
# of parameters, however some checks are omitted for simplicity.

def MetropolisDDM(k=10000):
    while True:
        n = int(input('Number of alternatives: '))
        if n <= 0:
            print('N must be a positive integer!')
        else:
            print('Your alternatives are 0,1,...,' + str(n-1))
            break
        
    u = np.arange(n)
    
    while True:
        u_options = input('Advanced utility options (0/1)? ')
        if int(u_options) in (0,1):
            break
        else:
            print('Please enter a valid input (0/1)')
        
    if int(u_options) == 1:
        print('Input utilities: ')
        u = [float(input('u(' + str(i) + ') = ')) for i in range(n)]
        u = np.array(u)
        
    u = u * 7.071 / (np.max(u)-np.min(u))
    
    while True:
        t = float(input('Time limit: '))
        if t > 1:
            break
        else:
            print('The time limit needs to be strictly greater than 1!')
        
    upper_barrier = np.log(t+1) / (np.max(u)-np.min(u))
        
    lower_barrier = upper_barrier
        
    while True:
        barrier_options = input('Advanced threshold settings (0/1)? ')
        if int(barrier_options) in (0,1):
            break
        else:
            print('Please enter a valid input (0/1)')
        
    if int(barrier_options) == 1:
        upper_barrier = float(input('Input acceptance threshold: '))
        lower_barrier = float(input('Input rejection threshold: '))
        
    while True:
        q_o = input('Advanced exploration settings (0/1)? ')
        if int(q_o) in (0,1):
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
    
    print()        
    print(n, 'alternatives with normalized utilities ', u, 'choose in ', t, 'time units')
    print()
    print('Acceptance/rejection thresholds: ', [upper_barrier,lower_barrier])
    print()
        
    if q_o: 
        print('Exploration matrix: ')
        print(em)
        print()
    else:
        print('Exploration matrix: ')
        print(explo_matrix_unif(n))
        print()
    
        
    p = np.exp(upper_barrier*u) / np.sum(np.exp(upper_barrier*u)) # softmax
    
    choice_count = np.zeros(n, dtype=int) # choice count vector
            
    # nc : number of choices from A = {0,...,D-1}

    for nc in range(k):
            
    ## DECISION PROCEDURE 
            
        s = 0 # clock
            
        b = np.random.randint(n) # first automatically accepted proposal 
            
        while True:
            if not(q_o):
                a = uniform_proposal(n,b)
            else:
                a = nonuniform_proposal(em,b)
                if a == b:
                    continue
                
            RT, CO = DDMcomparison(u[a], u[b], lower_barrier, upper_barrier)
            
            s += RT
            
            if s > t:
                break
            elif CO:
                b = a
                
        final_choice = b
        choice_count[final_choice] += 1
            
    choice_freq = choice_count / np.sum(choice_count)
            
    print('Choice count:', choice_count)
    print()
    print('Total variation distance:', np.round( np.linalg.norm(choice_freq - p, 1) / 2 , decimals = 5 ) )
    print()        
    print('Maximum simulation error:', np.round( np.max(np.abs(choice_freq - p)) , decimals = 5) )
        
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