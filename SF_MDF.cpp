// Include functions necessary for building code
#include<iostream>
#include<Eigen/Dense>
#include<vector>
#include <numeric>
#include <cmath>
#include <string>   
#include <sstream> 
#include <chrono>
#include <iomanip>
#include <fstream>
// #include <linalg>
#include <limits>

// Use namespace of various functions
using namespace std;
using namespace Eigen;

MatrixXd Hamiltonian(int l, double t, bool pbc, double v, double g, double u, double b, double q){
    /**
     * This function generates a generic Hamiltonian for 1D
     * non-interacting spinless fermions with NN hoppings 
     * and on-site potentials
     * 
     * INPUTS:
     * l: number of sites; integer
     * t: hopping parameter; float
     * pbc: periodic boundary conditions; true or false
     * v: constant on-site potential; float
     * g: random perturbation strength; float
     * u: trap strength; float
     * b: Bragg pulse strength; float
     * q: Bragg wavenumber; float
     * 
     * OUTPUTS: 
     * Hamiltonian: lxl matrix with correpsonding potentials and hoppings.
     */
    MatrixXd H(l,l);
    H.setZero();

    // NN Hoppings
    for (int i=0; i < l-1; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }
    // On-site terms
    for (int i=0; i<l; i++){
        H(i,i) += v; // constant on-site
        H(i,i) += g*((double)rand())/RAND_MAX; // random on-site
        H(i,i) += ((double)u*(i-l/2+0.5)*(i-l/2+0.5)/((l/2)*(l/2))); //trap potential
        H(i,i) += b*cos(q*i)*cos(q*i); //Bragg pulse
    }
    // Yields periodic boundary conditions
    if(pbc==true){
        H(0,l-1) = -t;
        H(l-1,0) = -t; 
    }
    return H;
}

void PrintHamiltonian(MatrixXd M, int L){
    /**
     * Given a Hamiltonian M of dimension L, print the components.
     * 
     * INPUTS:
     * M: matrix representation of Hamiltonian
     * L: dimension of matrix M
     * 
     * OUTPUTS:
     * Shows in the terminal the LxL representation of M. 
     */
    for(int i = 0; i < L; i++){
        for(int j = 0; j < L; j++){
            cout << M(i,j) << " \n"[j == L-1];
        }
    }
}

MatrixXcd OBDMZeroT(int N, MatrixXd P, MatrixXd wf2, ArrayXcd E2, double t){
    /**
     * Calculate one-body density matrix (OBDM) at some time t
     * for temperature T=0.
     * 
     * INPUTS:
     * L; number of lattice sites; integer
     * wf1: matrix of eigenvectors of pre-quench state; LxN matrix
     * wf2: matrix of eigenvectors of post-quech state; LxL matrix
     * E2: vector of eigenvalues of post-quench state; Lx1 vector
     * t: time in units of hbar/hopping; float
     * OUTPUTS:
     * C(t): LxL correlation matrix
     */
    complex<double> imag(0,t);
    DiagonalMatrix<complex<double>, 1000> D;
    D.diagonal() << (imag*E2).exp();
    MatrixXd P1 = (wf2.transpose())*P;
    MatrixXcd P2 = D*P1;
    MatrixXcd Pt = wf2*P2;
    return Pt * (Pt.adjoint());
}

namespace Eigen{
template<class Matrix>
void write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
    }
}

VectorXd nk(int L,  MatrixXd C){
    VectorXd sites(L);
    iota(sites.begin(),sites.end(),0);
    VectorXcd k(L/2 + 1);
    iota(k.begin(),k.end(),0);

}

int main() {
    const float pi=3.1415926535897932384;

    int L=1000, N=51;
    double t=1.0, v=0.0, g_rand=0.0, g_trap=0.0, g_Bragg=10.0, q=pi/4;
    MatrixXd Hpre = Hamiltonian(L,t,false,v,g_rand,g_trap,g_Bragg,q); 
    MatrixXd Hpost = Hamiltonian(L,t,false,v,g_rand,0.0,g_Bragg,q); 
    SelfAdjointEigenSolver<MatrixXd> es1(Hpre);
    SelfAdjointEigenSolver<MatrixXd> es2(Hpost);
    MatrixXd pre_state = es1.eigenvectors().leftCols(N), post_state = es2.eigenvectors();
    ArrayXcd post_eng = es2.eigenvalues().cast<complex<double>>().array();

    double time;
    int j=0;
    auto start = std::chrono::high_resolution_clock::now();
    // Multiple times
    for(int j=0; j<100; j++){
        // stringstream filename_buffer;
        time = 0.1 * j;
        MatrixXd C = OBDMZeroT(N,pre_state,post_state,post_eng,time).real();
        // filename_buffer << "Bragg_quench/C_free-bragg_L=1000_N=51_t=" <<  time << ".bin";
        // cout << filename_buffer.str() << endl;
        // write_binary(filename_buffer.str().c_str(),C); 
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Correlation matrix execution time: " << duration.count() << " seconds" << std::endl;


    return 0;
}