#include <random_kitchen_sinks.hpp>

// g++ -g -Wall -o sorf sorf_ex.cpp -lrandom_sinks -larmadillo

int main() {
    arma::mat X = arma::randu(1000, 200);

    double gamma = 1; int n_features = 30;
    random_sinks::SORF sorf(gamma, n_features);

    sorf.fit(X);
    arma::mat features = sorf.get_features(X);

    arma::mat x_new = arma::randu(1,200);
    arma::mat features_new = sorf.get_features(x_new);

    X.save(arma::hdf5_name("data.h5","X"));
    features.save(arma::hdf5_name("data.h5","feats"));

    return 0;
}