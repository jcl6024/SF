using LinearAlgebra
using Plots
using NLsolve
using Optim
using NonlinearSolve # For solving nonlinear equations
using Roots
using Distributed

function ZeroMat(N::Int)
    return zeros((N,N))
end

function TrapHamiltonian(L::Int, t::Float64, v::Float64, g::Float64)
    """
    Writes matrix representation of spinless fermion Hamiltonian on a lattice with L sites
    and a quadratic trapping potential of strength g. 

    INPUTS
    L: number of lattice sites
    t: hopping parameter
    v: constant on-site potential
    g: trapping potential strength

    OUTPUT
    H: LxL matrix representation of non-interacting fermion Hamiltonian
    """
    H = ZeroMat(L)
    for i in range(1,L)
        H[i,i] = v + g*(i-L/2)*(i-L/2)
    end
    for i in range(1,L-1)
        H[i,i+1] = -t
        H[i+1,i] = -t
    end
    return H
end

function FreeHamOBC(L::Int, t::Float64, v::Float64)
    """
    Writes matrix representation of spinless fermion Hamiltonian on a lattice with L sites
    and a quadratic trapping potential of strength g. 

    INPUTS
    L: number of lattice sites
    t: hopping parameter
    v: constant on-site potential
    g: trapping potential strength

    OUTPUT
    H: LxL matrix representation of non-interacting fermion Hamiltonian
    """
    H = ZeroMat(L)
    for i in range(1,L)
        H[i,i] = v 
    end
    for i in range(1,L-1)
        H[i,i+1] = -t
        H[i+1,i] = -t
    end
    return H
end

# function PartitionFunc(L, beta, E, mu)
#     """
#     """
#     return ones(L) + exp.(-beta*(E-mu*ones(L)))
# end

function NCorrFiniteT(L::Int, beta::Any, U::Any, E::Vector{Float64}, mu::Any)
    """
    Calculate N-body correlation function in equilibrium at finite temperature

    INPUTS
    L: number of sites
    beta: inverse temperature
    U: matrix of eigenvectors
    E: vector of eigenvalues
    mu: chemical potential

    OUTPUT
    C: matrix of one-body correlations
    """
    identity = Matrix(1.0I,L,L)
    D = inv(identity + Diagonal(exp.(-beta * (E - mu*ones(L)))))
    Dprime = transpose(U) * D * U
    return identity - Dprime
end

function GetChemicalPotential(L::Int, beta::Any, N::Int, U::Any, E::Vector{Float64})
    """
    Get chemical potential provided N particles at temperature T

    INPUTS
    L: number of lattice sites; int
    N: number of particles; int
    beta: inverse temperature; float
    U: matrix of eigenvectors; matrix of floats
    E: list of eigenvalues; list of floats

    OUTPUT
    mu: chemical potental; float
    """
    func(mu) = sum(diag(NCorrFiniteT(L,beta,U,E,mu))) - N
    mu_guess = maximum(E[1:N])
    sol = find_zero(func,mu_guess)
    return sol
end

function pij(t::Any,beta::Any,mu::Any,L::Int,U1::Any,U2::Any,E1::Vector{Float64},E2::Vector{Float64})
    """
    Calculate equal-time one-body density matrix at finite T and non-zero time after a quench.
    
    INPUTS
    t: time; units of hbar/J
    beta: inverse temperature; units of 1/J
    mu: chemical potential; units of J
    L: lattice sites; dimensionless
    U1: matrix of eigenvectors of pre-quench Hamiltonian
    U2: matrix of eigenvectors of post-quench Hamiltonian
    E1: vector of eigenvalues of pre-quench Hamiltonian
    E2: vector of eigenvalues of post-quench Hamiltonian

    OUTPUT
    One-body correlation density matrix.
    """
    C0 = NCorrFiniteT(L,beta,U1,E1,mu)
    D = Diagonal(exp.(complex(0,t)*E2))
    D1 = D * U2
    D2 = transpose(conj(U2)) * D1
    C1 = C0 * D2
    C2 = conj(D2) * C1
    return C2
end

function nkt(k::Any, L::Int, beta::Int, t::Any, P::Any)
    """
    Calculate momentum distribution function at finite T and non-zero time after a quench.
    
    INPUTS
    k: quasi-momentum; units of 1/a
    L: lattice sites; dimensionless
    beta: inverse temperature; units of 1/J
    t: time; units of hbar/J
    U1: matrix of eigenvectors of pre-quench Hamiltonian
    U2: matrix of eigenvectors of post-quench Hamiltonian
    E1: vector of eigenvalues of pre-quench Hamiltonian
    E2: vector of eigenvalues of post-quench Hamiltonian
    mu: chemical potential; units of J

    OUTPUT
    Momentum distribution function
    """
    sites = range(0,999,length=1000)
    eik = exp.(k*sites*complex(0,-1)) * transpose(exp.(k*sites*complex(0,1))/L)
    return sum(diag(P*eik))
end