# generate the slater determinant hat{a}^dagger_1hat{a}^dagger_2...hat{a}^dagger_tau ket{0}
import numpy as np
from scipy.stats import unitary_group
from numpy import loadtxt
from numpy import savetxt
import itertools
from numpy import loadtxt
from numpy import savetxt


def Gen_randState(d):
    phi = np.random.random(d) + np.random.random(d)*1.j
    norm = np.sqrt(sum(np.square(np.abs(phi))))
    phi = phi/norm
    return phi
####generate random input quantum state phi for slater determinant
# n = 7
# d = 2**n
# phi = Gen_randState(d)
# print(phi)
# savetxt('Slater_phi'+str(n) + '.txt', phi,delimiter=',')

def gen_InputState(n,phi):
    state_0 = np.zeros(2**n)
    state_0[0] = 1
    phi_anc = state_0 + np.kron(np.array([0,1]), phi)
    rho = np.outer(phi_anc,np.conj(phi_anc))
    rho /= 2
    return rho

phi = loadtxt('Slater_phi7.txt',dtype='complex')
n = 8
rho = gen_InputState(n,phi)
###print(rho)
#np.savetxt('Slater_InputState.txt', rho, delimiter=',')

def ind(S):
    """given a list of integers >=1, S, returns a list with all entries shifted by -1"""
    
    return [(s-1) for s in S]


def subsets(n,k):
    """returns all subsets of {1,...,n} of cardinality k"""
    
    return list(itertools.combinations(np.arange(1,n+1), k))

def matching_slater(n,S, tau):
###return the number representation 1-S for n-bitstring, where the length of S is tau, e.g.S = [1,2], n = 3, numb_b = 1110=11
    
    s_res = np.zeros(n+1)
    s_res[0] = 1
    for j in range(tau):
        s_res[S[j]] = 1
  #  print('set converted',s_res)
    #convert s_res into a number
    num_b = 0
    temp = 1
    for j in range(n+1):
        num_b = num_b + temp * s_res[n-j]
        temp *= 2
    return int(num_b)


def matching_short_slater(n,S,tau):
###return the number representation S for n-bitstring, where the length of S is tau, e.g.S = [1,2], n = 3, numb_b = 110=3
    s_res = np.zeros(n)
    for j in range(tau):
        s_res[S[j]-1] = 1
   # print('set converted',s_res)
    #convert s_res into a number
    num_b = 0
    temp = 1
    for j in range(n):
        num_b = num_b + temp * s_res[n-1-j]
        temp *= 2
    return int(num_b)

def Verify_Slater(n,psi, tau,V):
    # the size of V equals n by n.
    sets = subsets(n,tau)
    res = 0
    Stau = np.array([j+1 for j in range(tau)])
    for S in sets:
        bs = matching_short_slater(n,S,tau)
    #    print('S:',S, 'phi',bs,'=',psi[bs], np.linalg.det(np.conj(V)[np.ix_(ind(Stau), ind(S))]))
        res += np.linalg.det(np.conj(V)[np.ix_(ind(Stau), ind(S))]) * np.conj(psi[bs])
    return res
#generate the 2n by 2n matrix for slater determinant, associated with V.
def Gen_QSlater(V,n):
    
    Q = np.zeros((2*n, 2*n))
    
    for j in range(n):
        for k in range(n):
            Q[2 * j ][2 * k ] = V[j][k].real
            Q[2 * j ][2 * k + 1] = -1 * V[j][k].imag
            Q[2 * j + 1][2 * k ] = V[j][k].imag
            Q[2 * j + 1][2 * k + 1] = V[j][k].real
    return Q

#note that here we can only use V, the Q explanation in Wan's paper does not work!!!
def gen_observable(V,n, tau): ###the first column of the observable is coef.
    
    sets = subsets(n,tau)
    
    #l_sets = len(sets)
    
    #coefs = np.zeros(2**(n+1), dtype='complex128')
    coefs = np.zeros(2**n, dtype='complex128')
    Stau = np.array([j+1 for j in range(tau)])
    for S in sets:
        #bs = matching_slater(n,S,tau)#int type
        bs = matching_short_slater(n, S, tau)
       # print(S)
        coefs[bs] = np.linalg.det(np.conj(V)[np.ix_(ind(Stau), ind(S))])
    
    return coefs

# n = 7
# tau = 6
# #####generate n-1 by n-1 unitary V
# V = unitary_group.rvs(n)
# Q = Gen_QSlater(V,n)
# coefs = gen_observable(V,n,tau)
# savetxt('Slater_vector'+ str(n) +'.txt', coefs, delimiter=',')  
# print(len(coefs))

def Gen_SlaterHam(Nq):
    Dim = 2**Nq
    coefs = loadtxt("Slater_vector7.txt", dtype='complex')
    tau_state = np.kron(np.array([0,1]), coefs)
    zero = np.zeros(Dim)
    zero[0] = 1
    Ham = np.outer(tau_state, zero)
    return Ham

##generate 8-qubit Hamiltonian
Nq = 8
Ham = Gen_SlaterHam(Nq)
savetxt('Slater_Ham'+ str(Nq) +'.txt', Ham, delimiter=',')  
