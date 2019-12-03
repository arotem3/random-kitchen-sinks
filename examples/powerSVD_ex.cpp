#include <random_kitchen_sinks.hpp>

// g++ -g -Wall -o powerSVD_ex powerSVD_ex.cpp -lrandom_sinks -larmadillo

int main() {
    arma::mat X = arma::orth(arma::randn(5000, 5)) * arma::diagmat(arma::vec({100,10,1,0.1,0.01})) * arma::randn(5, 1000); // actual rank = 5, singular values decay exponentially
    arma::mat U, V;
    arma::vec s;
    double err = random_sinks::powerSVD(U, s, V, X, 3, 10);

    std::cout << "singular value error estimate: " << err << "\n"
              << "total error: " << arma::norm(X - U*arma::diagmat(s)*V.t(), "fro") << "\n"
              << "covariance eigenvalues: " << s.t() << "\n";
    return 0;
}