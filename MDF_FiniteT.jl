using LinearAlgebra
using Plots
using NLsolve
using Optim
using NonlinearSolve 
using Roots
using Distributed

#############################
### Hamiltonian Functions ###
#############################
function ZeroMat(N::Int)
    return zeros((N,N))
end

function TrapHamiltonian(L::Int, t::Float64, v::Float64, g::Float64, pbc::Bool)
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
    if pbc == true
        H[L,1] = -t
        H[1,L] = -t
    end
    return H
end

function BraggHamiltonian(L::Int, t::Float64, v::Float64, g::Float64, m::Int, q::Float64, pbc::Bool)
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
        H[i,i] = v + g*(i-L/2)*(i-L/2) + m *cos.(q*(i-L/2)).^2
    end
    for i in range(1,L-1)
        H[i,i+1] = -t
        H[i+1,i] = -t
    end
    if pbc == true
        H[L,1] = -t
        H[1,L] = -t
    end
    return H
end

function FreeHamiltonian(L::Int, t::Float64, v::Float64,pbc::Bool)
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
    if pbc == true
        H[L,1] = -t
        H[1,L] = -t
    end
    return H
end


###############################
### One body density Matrix ###
###############################
function NCorrFiniteT(L::Int, beta::Float64, U::Matrix{Float64}, E::Vector{Float64}, mu::Float64)
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
    identity::Matrix{Float64} = Matrix(1.0I,L,L)
    D::Matrix{Float64} = inv(identity + Diagonal(exp.(-beta * (E - mu*ones(L)))))
    Dprime::Matrix{Float64} = U * D * transpose(U)
    # Dprime2::Matrix{Float64} = U * Dprime
    C::Matrix{Float64} = identity - Dprime
    return Symmetric(C)
end

function GetChemicalPotential(L::Int64, beta::Float64, N::Int64, U::Matrix{Float64}, E::Vector{Float64})
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

function pij(t::Float64,U2::Matrix{Float64},E2::Vector{Float64},C0::Matrix{Float64})
    """
    Calculate equal-time one-body density matrix at non-zero time after a quench.
    
    INPUTS
    t: time; units of hbar/J
    beta: inverse temperature; units of 1/J
    mu: chemical potential; units of J
    L: lattice sites; dimensionless
    U2: matrix of eigenvectors of post-quench Hamiltonian
    E2: vector of eigenvalues of post-quench Hamiltonian
    C0: Initial one-body correlation function, can be zero T or finite T.

    OUTPUT
    One-body correlation density matrix.
    """
    D = Diagonal(exp.(complex(0,t)*E2))
    D1 = U2 * D * transpose(U2)
    C2 = adjoint(D1) * C0 * D1
    return Symmetric(C2)
end

function nkt(k::Float64, L::Int64, P::Matrix{Float64}, sites::Vector{Float64})
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
    A = exp.(sites*complex(0,k))
    eik = A * transpose(conj(A)) 
    return sum(eik .* P)/L
end

#################
### load data ###
#################
L::Int64 = 1000

### pre quench ###
E::Vector{Float64} = eigvals(FreeHamiltonian(L,1.0,0.1,true))
U::Matrix{Float64} = eigvecs(FreeHamiltonian(L,1.0,0.1,true))
# E::Vector{Float64} = eigvals(TrapHamiltonian(L,1.0,0.1,2e-5,false))
# U::Matrix{Float64} = eigvecs(TrapHamiltonian(L,1.0,0.1,2e-5,false))

### post quench ###
E2::Vector{Float64} = eigvals(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,true))
U2::Matrix{Float64} = eigvecs(BraggHamiltonian(L,1.0,0.1,0.0,20,pi/4,true))
sites::Array{Float64,1} = range(0,L-1,length=L);



#####################
### Calculate MDF ###
#####################
tpulse::Float64 = 0.1;
Np::Int64 = 31;
Temps::Array{Float64} = [2.0,10.0,100.0]
for beta in Temps
    @time mu::Float64 = GetChemicalPotential(L, beta, Np, U, E)
    @time C::Matrix{Float64} = NCorrFiniteT(L, beta, U, E, mu)
    @time p_pulse::Matrix{Float64} = real(pij(tpulse,U2,E2,C));
    for t in range(0,50,2)
        p::Matrix{Float64} = real(pij(t,U,E,p_pulse));
        @time n::Vector{Float64} = real([nkt(q,L,p,sites) for q in range(0,pi,501)]);
        open(string("Bragg_quench/n_free_L=1000_N=31_t=",t,"T=",beta,".bin"),"w") do f
            write(f,n)
        end
        println(string("t=",t," done!"))
    end
end