#include<iostream>
#include<Eigen/Dense>
#include <chrono>

using namespace std;
using namespace Eigen;

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    double t, v;

    t=1;
    v=0.1;

    MatrixXd H(100,100);
    H.setZero();

    for (int i=0; i<100; i++){
        // H(i,i+1) = -t;
        // H(i+1,i) = -t;
        H(i,i) = v;
    };

    EigenSolver<MatrixXd> es(H);

    //cout<<endl<<endl<<H<<endl<<endl;
    cout<< es.eigenvalues() <<endl<<endl;
    // cout<< es.eigenvectors()<<endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}