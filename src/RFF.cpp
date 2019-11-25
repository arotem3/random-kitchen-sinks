#include <random_kitchen_sinks.hpp>

/* RFF(gamma=1, n_feats=100, seed=-1) : initialize random fourier features.
 * --- gamma : parameter of Gaussian kernel, i.e. exp(-gamma * ||x||^2)
 * --- n_feats : number of random features
 * --- seed : random seed, if seed<0 then the seed will be randomized. */
random_sinks::RFF::RFF(double gamma, int n_feats, int seed) : weights(_w), bias(_b) {
    if (gamma <= 0 || n_feats <= 1) {
        std::string excpt = "RFF() error: invalid parameter values.\n\tgamma = " + std::to_string(gamma) + " (must be > 0)\n\tn_feats = " + std::to_string(n_feats) + " (must be > 1...much greater...)\n";
        throw std::runtime_error(excpt);
    }
    _nfs = n_feats;
    _gamma = gamma;
    if (seed < 0) arma::arma_rng::set_seed_random();
    else arma::arma_rng::set_seed(seed);
}

/* fit(x) : produce the coefficient matrix.
 * --- x : matrix whose dimensions we match. */
void random_sinks::RFF::fit(const arma::mat& X) {
    _dim = X.n_cols;
    _w = 2*_gamma*_dim*arma::randn(_dim,_nfs);
    _b = arma::randu<arma::rowvec>(_nfs) * 2*M_PI;
}

/* get_features(x) : evaluate random features.
 * --- x : matrix to evaluate features for. */
arma::mat random_sinks::RFF::get_features(const arma::mat& X) {
    if (X.n_cols != _dim) {
        std::string excpt = "RFF::get_features() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_dim) + ").\n";
        throw std::runtime_error(excpt);
    }
    arma::mat Y = X * _w;
    Y.each_row() += _b;
    Y = arma::cos(Y);
    return Y;
}