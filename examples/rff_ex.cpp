#include <random_kitchen_sinks.hpp>

// g++ -g -Wall -o rff rff_ex.cpp -lrandom_sinks -larmadillo

int main() {
    arma::mat X = arma::randu(1000, 200);

    double gamma = 1; int n_features = 30;
    random_sinks::RFF rff(gamma, n_features);

    rff.fit(X);
    arma::mat features = rff.get_features(X);

    arma::mat x_new = arma::randu(1,200);
    arma::mat features_new = rff.get_features(x_new);

    X.save(arma::hdf5_name("data.h5","X"));
    features.save(arma::hdf5_name("data.h5","feats"));

    return 0;
}