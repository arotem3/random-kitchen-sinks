#include <random_kitchen_sinks.hpp>

// g++ -g -Wall -o nystrom_ex nystrom_ex.cpp -lrandom_sinks -larmadillo

int main() {
    arma::mat X = arma::randu(1000, 200);

    double gamma = 1; int n_features = 30;
    std::string method = "powerSVD"; // uses random_sinks::powerSVD
    // std::string method = "arma"; // uses arma::svd
    random_sinks::gauss_nystrom nystrom(n_features, gamma);
    nystrom.fit(X, method);
    arma::mat features = nystrom.get_features(X);
    arma::mat x_new = arma::randu(1,200);
    arma::mat features_new = nystrom.get_features(x_new);
    
    X.save(arma::hdf5_name("features.h5","X"));
    features.save(arma::hdf5_name("features.h5","feats"));

    return 0;
}