#include <random_kitchen_sinks.hpp>

// g++ -g -Wall -o rSVD_ex rSVD_ex.cpp -lrandom_sinks -larmadillo

int main() {
    arma::arma_rng::set_seed_random();
    arma::mat X = arma::orth(arma::randn(5000, 5)) * arma::diagmat(arma::vec({100,10,1,0.1,0.01})) * arma::randu(5, 1000); // actual rank = 5, singular values decay exponentially
    arma::mat U,V;
    arma::vec s;
    random_sinks::rSVD(U,s,V,X,3);
    
    std::cout << "total error: " << arma::norm(X - U*arma::diagmat(s)*V.t(), "fro") << "\n"
              << "estimated singular values: " << s.t() << "\n";
    return 0;
}