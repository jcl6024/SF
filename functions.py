#########################
### import statements ###
#########################
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle
import matplotlib.animation as animation
from scipy.optimize import fsolve

#########################
### Analytic Spectrum ###
#########################
def ODFFHam(t,v,k):
    """
    Exact single particle spectrum.
    """
    return - 2 * t * np.cos( k ) + v

#########################################################################
### Equal-time one-body density matrix; equlibrium & zero temperature ###
#########################################################################
def C(i,j,k,P):
    '''
    Single particle pure state correlations. Essentially computes overlap between
    wavefunctions of particles at sites i and j.

    i,j: site indices; integer
    k: eigenvalue index; integer
    P: matrix of eigenstate coefficients
    '''
    C = P[k][j] * P[k][i]
    return C

def NC(i,j,nk,P):
    '''
    Calculates N-particle correlation function using single-particle pure state corre-
    lation function defined above. 

    i,j: site indices; integer
    nk: list of 0's and 1's; represents eigenstate in energy occupation basis;
        ordered according to that representation
    P: matrix of eigenstate coefficients
    '''
    NC = []
    k = 0
    for n in nk:
        NC.append(n*C(i,j,k,P))
        k += 1
    return sum(NC)

def NC2(nk,U):
    """ 
    Calculates total correlation matrix for a given initial state specified by an occupation
    representation nk. 

    nk: list of 0's and 1's; represents eigenstate in energy occupation basis;
        ordered according to that representation. MUST be a numpy array.
    U: matrix of eigenstate coefficients; matrix of eigenvectors
    """
    idx = np.where(nk == 1)
    P = U[idx]
    return np.matmul(P.T,P)


############################################################################
### Equal-time one-body density matrix; nonequlibrium & zero temperature ###
############################################################################
def SPOverlap2(wf1,mode1,wf2,mode2):
    """
    L: number of sites
    wf1,wf2: LxL matrix of eigenvectors; also called U matrix
    mode1,mode2: relevant eigenenergy mode number
    """
    return np.sum(np.conj(wf1[mode1]) * wf2[mode2])
    

def SlaterMatN(N,qs,gs):
    """
    N: number of particles; if L is even, this should be odd
        and vice versa--avoids degenerate ground states. This
        is necessary when using numeric eigenstates.
    qs: quenched state; list of L entries
    gs: ground state; list of L entries
    mode: list of N integers corresponding tot e
    """
    S = np.zeros((N,N),dtype=np.float64)
    for i in range(N):
        for j in range(N):
            S[i,j] = SPOverlap2(qs,j,gs,i)
    
    return S

def GaussianMat(N,wf1,wf2,E2,t):
    """
    Construct P matrix of Guassian states.

    N: number of particles
    wf1: matrix of pre-quench states, should be a row matrix
    wf2: matrix of post-quench states, should be a row matrix
    E2: array of post-quench energy eigenvalues
    t: time

    NOTE: information about the total number of sites L should be 
    contained within wf1,wf2, and E2. 
    """
    P = wf1[:N].T # we want P to be a column matrix, hence .T
    U = wf2.T
    z = E2*complex(0,t) # np.array([complex(0,E2[j]*t) for j in range(len(E2))])
    D = np.diag(np.exp(z))
    P1 = np.matmul(np.conj(U.T),P)
    P2 = np.matmul(D,P1)
    P3 = np.matmul(U,P2)
    return P3

def NCorrMat(P):
    """
    Calculatte the N-body correlation matrix by exploiting the 
    properties of Slater determinants and fermionic gaussian
    states.

    P: GaussianMat(N,sf1,wf2,E2,t)
    """
    Correlation = np.matmul(P,np.conj(P.T))
    return Correlation



###########################################################################
### Equal-time one-body density matrix; equlibrium & finite temperature ###
###########################################################################
def FermiEnergy(E,N):
    """
    Calculate Fermi energy of N non-interacting spinless fermions on a lattice of
    L sites. 

    E: list of L eigenvalues
    N: number of fermions; N<L
    """
    return np.sum(E[:N])


def PartitionFunc(L,beta,E,mu):
    """ 
    Calculate N-body correlation function in equilibrium at a finite temperature.

    L: size of lattice; integer
    beta: inverse temperature; float
    E: List of L eigenvalues
    mu: chemical potential; float
    """
    return np.prod(np.ones(L)+np.exp(-beta*(E-mu)))

# def PartitionFunc2(L,beta,E,mu):
#     """ 
#     Calculate N-body correlation function in equilibrium at a finite temperature.

#     L: size of lattice; integer
#     beta: inverse temperature; float
#     E: List of L eigenvalues
#     mu: chemical potential; float
#     """
#     I = np.identity(L)
#     return np.linalg.det(I+np.diag(np.exp(-beta*(E-mu))))

def NCorrFiniteT(i,j,L,beta,U,E,mu):
    """ 
    Calculate N-body correlation function in equilibrium at a finite temperature.

    i,j: site indices; integer
    L: size of lattice; integer
    beta: inverse temperature; float
    U: LxL diagonalization transformation matrix from lattice-site to energy basis
    E: List of L eigenvalues
    mu: chemical potential; float
    """
    Z = PartitionFunc(L,beta,E,mu)
    Delta = np.identity(L)
    Delta[i][j] = 1
    I = np.identity(L)
    D = np.diag(np.exp(-beta*(E-mu)))
    D1 = np.matmul(D,np.conj(U.T))
    D2 = np.matmul(U,D1)
    A = I + np.matmul(Delta,D2)
    return np.linalg.det(A) / Z - 1

def NCorrFiniteTDiag(L,beta,U,E,mu):
    """ 
    Calculate N-body correlation function in equilibrium at a finite temperature.

    i,j: site indices; integer
    L: size of lattice; integer
    beta: inverse temperature; float
    U: LxL diagonalization transformation matrix from lattice-site to energy basis
    E: List of L eigenvalues
    mu: chemical potential; float
    """
    I = np.identity(L)
    D = np.linalg.inv(I + np.diag(np.exp(-beta*(E-mu))))
    D1 = np.matmul(D,U)
    D2 = np.matmul(np.conj(U.T),D1)
    return I - D2

def GetChemicalPotential(L,N,beta,U,E):
    """ 
    Returns best fit for chemical potential given a temperature beta.
    NOTE: This function is not fast.

    L: number of sites
    N: total number of particles
    beta: inverse temperature
    U: unitary diagonalization matrix
    E: List of L eigenvalues
    """
    func = lambda mu: np.sum(np.diag(NCorrFiniteTDiag(L,beta,U,E,mu))) - N
    mu_guess = max(E[:N])
    return fsolve(func,mu_guess)

def FermiDirac(beta,mu,t,v,k):
    ''' 
    Thermal distribution for fermions on a lattive with chemical potential mu. 

    beta: inverse temperature
    mu: chemical potential
    t: hopping parameter
    v: constant on-site potential
    k: quasi-momentum
    '''
    return 1/(np.exp(beta*(ODFFHam(t,v,k)-mu))+1)

def nk(k,L,NCorr):
    ''' 
    Calculate local number density as a function of quasi-momentum.

    k: quasi-momentum number.
    i,j: site indices; integer
    L: size of lattice; integer
    beta: inverse temperature; float
    U: LxL diagonalization transformation matrix from lattice-site to energy basis
    E: List of L eigenvalues
    mu: chemical potential; float
    '''
    sites = np.linspace(0,L-1,L)
    dumvec = L**(-1/2)*np.exp(k*sites*complex(0,1)).reshape(1,L)
    eik = np.matmul(np.conj(dumvec.T),dumvec)
    nk = np.sum(eik*NCorr)
    return nk 



##############################################################################
### Equal-time one-body density matrix; nonequlibrium & finite temperature ###
##############################################################################
def pij(t,T,mu,L,U1,U2,E1,E2):
    """
    Calculate equal-time one-body density matrix at a time t after a quench
    for a system initially at a finite temperature inv(T).

    t: time; float; units of hbar/J 
    T: inverse temperature; float; units of 1/J
    mu: pre-quench chemical potental; float; units of J
    L: number of lattice sites; integer; dimensionless
    U1: matrix of pre-quench eigenvectors; LxL array of floats; dimensionless
    U2: matrix of post-quench eigenvectors; LxL array of floats; dimensionless
    E1: array of pre-quench eigenvalues; Lx1 array of floats; units of J
    E2: array of pre-quench eigenvalues; Lx1 array of floats; units of J
    """
    C0 = NCorrFiniteTDiag(L,T,U1,E1,mu)
    D = np.diag(np.exp(complex(0,1)*E2*t))
    D1 = np.matmul(D,U2)
    D2 = np.matmul(np.conj(U2).T,D1)
    C1 = np.matmul(C0,D2)
    C2 = np.matmul(np.conj(D2),C1)
    return C2

# def eik(L):
#     """ 
#     Makes a ((L,L),L) tensor which contains all possible exp(ik(i-j)) for all L values of k 
#     given k lattice sites.
#     """
#     sites = np.linspace(0,L-1,L)
#     eik = []
#     for k in np.linspace(-np.pi,np.pi,L+1):
#         eik.append(np.matmul(np.conj(L**(-1/2)*np.exp(k*sites*complex(0,1)).reshape(1,L).T),L**(-1/2)*np.exp(k*sites*complex(0,1)).reshape(1,L)))
#     return eik

def nkt(k,L,NCorr):
    ''' 
    Calculate local number density as a function of quasi-momentum, temperature, and time.

    k: quasi-momentum number.
    i,j: site indices; integer
    L: size of lattice; integer
    beta: inverse temperature; float
    time: time; float; units of hbar/J
    U1: matrix of pre-quench eigenvectors; LxL array of floats; dimensionless
    E1: array of pre-quench eigenvalues; Lx1 array of floats; units of J
    U2: matrix of post-quench eigenvectors; LxL array of floats; dimensionless
    E2: array of pre-quench eigenvalues; Lx1 array of floats; units of J
    mu: chemical potential; float
    '''
    sites = np.linspace(0,L-1,L)
    dumvec = L**(-1/2)*np.exp(k*sites*complex(0,1)).reshape(1,L)
    eik = np.matmul(np.conj(dumvec.T),dumvec)
    nk = np.sum(eik*NCorr)
    return nk