#include<iostream>
#include<Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <limits>
#include <fstream>
#include <cmath>

using namespace std;
using namespace Eigen;

MatrixXd RandHamiltonian(int num, double t, double v, double g){
    MatrixXd H(num,num);
    H.setZero();

    for (int i = 0; i<num; i++){
        H(i, i) = v + g * ((double)rand()) / RAND_MAX;
    }

    for (int i=0; i<num-1; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }

    H(0,num-1) = -t;
    H(num-1,0) = -t;

    return H;

    // EigenSolver<MatrixXd> es(H);

}

MatrixXd ShiftHamiltonian(int num, double t, double v, int g){
    MatrixXd H(num,num);
    H.setZero();

    for (int i = 0; i<num; i++){
        H(i, i) = v;
        if(i >= num/2) {
            H(i,i) += g;
        }
    }

    for (int i=0; i<num-1; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }

    H(0,num-1) = -t;
    H(num-1,0) = -t;

    return H;

}

MatrixXd TrapHamiltonian(int num, double t, double v, double g){
    // Writes Hamiltonian for free fermions on a lattice of L sites with 
    // a quadratic trap
    MatrixXd H(num,num);
    H.setZero();

    for (int i = 0; i<num; i++){
        H(i, i) = v + ((double)g*(i-num/2+0.5)*(i-num/2+0.5));
    }
    for (int i=0; i<num-1; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }

    // Uncomment/comment below for PBC/OBC
    H(0,num-1) = -t;
    H(num-1,0) = -t;

    return H;

}

MatrixXd BraggHamiltonian(int num, double t, double v, double g, int m, double q){
    // Writes Hamiltonian for free fermions on a lattice of L sites with 
    // a quadratic trap
    MatrixXd H(num,num);
    H.setZero();

    for (int i = 0; i<num; i++){
        H(i, i) = v + ((double)g*(i-num/2+0.5)*(i-num/2+0.5)) + m*cos(q*i)*cos(q*i) ;
    }
    for (int i=0; i<num-1; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }

    // Uncomment/comment below for PBC/OBC
    H(0,num-1) = -t;
    H(num-1,0) = -t;

    return H;

}

// VectorXd sites(L);
//     VectorXcd k(L);
//     complex<double> dk(0,(2*pi/L));
//     iota(sites.begin(),sites.end(),0);
//     iota(k.begin(),k.end(),-L/2);
//     // for (complex i: (dk*k).array().exp()) cout << i << ' ';
//     MatrixXcd D = (((dk*k).array()).exp()).matrix().asDiagonal();
//     PrintHamiltonian(D,L);

MatrixXi testmatrix(int Num){
    MatrixXi H(Num,Num);

    for(int i = 0; i<Num; i++){
        for(int j = 0; j < Num; j++){
            H(i,j) = i + j;
        }
    }

    return H;
}

int main() {
    // The following times how long the function takes to run
    auto start = std::chrono::high_resolution_clock::now();

    int num = 1000;
    double t = 1, v = 0, g = 2 * 0.000004, q = 0.1*3.14159265359;
    MatrixXd H;

    // H = TrapHamiltonian(num,t,v,0.0);
    H = BraggHamiltonian(num,t,v,g,5,q);

    // prints Hamiltonian for viewing
    // for(int i = 0; i < num; i++){
    //     for(int j = 0; j < num; j++){
    //         cout << H(i,j) << " \n"[j == num - 1];
    //     }
    // }

    SelfAdjointEigenSolver<MatrixXd> es(H);

    ofstream file;

    file.open("Bragg_quench/bragg_eigval_t=1_L=1000_q=pi10_PBC.txt");
    file << es.eigenvalues(); 
    file.close();

    file.open("Bragg_quench/bragg_eigvec_t=1_L=1000_q=pi10_PBC.txt");
    file << es.eigenvectors(); 
    file.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    // cout << fixed << setprecision(numeric_limits<double>::max_digits10) << es.eigenvalues();

    return 0;

}