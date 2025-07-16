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

##########################
### correlation matrix ###
##########################
function RamanNathC(L::Int, N::Int, A::Float64, k::Float64, U::Matrix{Float64})
    """
    Calculate 
    """
    D::Matrix{ComplexF64} = Diagonal(exp.(complex(0,-A)*cos.(k*range(0,L-1,L)).^2))
    return D
end

function HeavisideTheta(x)
    if x>0
        return 1
    else
        return 0
    end
end

function f(theta,thetaF,k,A)
    
end
