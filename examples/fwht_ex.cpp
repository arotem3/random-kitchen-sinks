#include <random_kitchen_sinks.hpp>

// g++ -g -Wall -o fwht fwhat_ex.cpp -lrandom_sinks -larmadillo

int main() {
    arma::arma_rng::set_seed(1223);

    arma::mat x = arma::randu(10,32);
    arma::mat xh = random_sinks::fwht(x);
    
    x.raw_print("x:");
    xh.raw_print("Hx:");
    
    return 0;
}