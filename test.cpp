#include<iostream>
#include<Eigen/Dense>
#include <chrono>
#include <fstream>

using namespace std;
using namespace Eigen;

// int main()
// {
//     int x;
//     int sum;
    
//     cout << "Type a number: ";
//     cin >> x;
//     sum = x * 10; 
//     cout << "That number multiplied by 10: ";
//     cout << sum;
// }

//Average grade in quantum mechanics calculator

// int main(){
//     int num_of_exams;
//     float sum=0, grade, weight;

//     cout << "Number of exams: ";
//     cin >> num_of_exams;

//     for (int i=1; i <= num_of_exams; i++) {
//         cout << "Grade on exam " << i << ": ";
//         cin >> grade;
//         cout << "Exam " << i << " weight: ";
//         cin >> weight;
//         sum += grade*weight;
//     }

//     cout << "Grade (equally weighted): " << sum;
// }

void random01()
{
    // Using srand() with time(0) to change
    // the random values everytime
    srand(time(0));

    cout << ((double)rand()) / RAND_MAX << endl;
};

void Hamiltonian(int num, float t, float v){
    MatrixXf H(num,num);
    H.setZero();

    for (int i = 0; i<num; i++){
        H(i, i) = v;
    }

    for (int i=0; i<num-1; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }

    // EigenSolver<MatrixXd> es(H);

}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    MatrixXd H(100,100);
    H.setZero();

    // for (int i=0; i<100; i++){
    //     for(int j=0; j<100; j++){
    //         H(i,j) = ((double)rand()) / RAND_MAX;
    //     };

    // };

    double t, v;
    t=1;
    v=0.1;

    for (int i = 0; i<100; i++){
        H(i, i) = v;
    }

    for (int i=0; i<99; i++){
        H(i,i+1) = -t;
        H(i+1,i) = -t;
    }

    EigenSolver<MatrixXd> es(H);

    //cout<<endl<<endl<<H<<endl<<endl;
    cout<< es.eigenvalues() <<endl<<endl;
    //cout<< es.eigenvectors()<<endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    ofstream file;
    file.open("test_eigenvalues.csv");
    file << es.eigenvalues();
    file.close();

    file.open("test_eigenvectors.csv");
    file << es.eigenvectors();
    file.close();

    return 0;
}
