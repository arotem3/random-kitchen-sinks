#include<random_kitchen_sinks.hpp>

/* gauss_nystrom(n_feats=128, gamma=1, seed=-1) : initialize gauss nystrom kernel pca object.
 * --- n_feats : number of principle components to compute.
 * --- gamma : parameter of Gaussian kernel, i.e. exp(-gamma * ||x||^2)
 * --- seed : random seed, if seed<0 then the seed will be randomized. */
random_sinks::gauss_nystrom::gauss_nystrom(uint n_feats, double gamma, int seed) : kernel_singular_vals(_s), kernel_left_vecs(_U), kernel_right_vecs(_V) {
    if (gamma <= 0 || n_feats <= 1) {
        std::string excpt = "gauss_nystrom() error: invalid parameter values.\n\tgamma = " + std::to_string(gamma) + " (must be > 0)\n\tn_feats = " + std::to_string(n_feats) + " (must be > 1...much greater...)\n";
        throw std::runtime_error(excpt);
    }
    _nfs = n_feats;
    _gamma = gamma;
    if (seed < 0) arma::arma_rng::set_seed_random();
    else arma::arma_rng::set_seed(seed);
}

/* fit(X, method="powerSVD", max_iter=10, tol=1e-8) : fit kernel pca.
 * --- X : data matrix.
 * --- max_iter, tol : if powerSVD is being used, these parameters are passed along to it, otherwise they are ignored.
 * --- method : method to compute PCA with, either "arma" or "powerSVD" */
void random_sinks::gauss_nystrom::fit(arma::mat X, std::string method, uint max_iter, double tol) {
    if (_nfs >= X.n_rows) {
        std::string excpt = "guass_nystrom::fit() error: number of components requested (=" + std::to_string(_nfs) + ") exceeds number of observations in X (X.n_rows=" + std::to_string(X.n_rows) + ")\n";
        throw std::runtime_error(excpt);
    }
    bool use_arma_svd = false;
    if (method == "arma") use_arma_svd = true;
    else if (method != "powerSVD") {
        std::string excpt = "gauss_nystrom::fit() error: method must be one of {\"arma\", \"powerSVD\"}. Method requested was \"" + method + "\".\n";
        throw std::runtime_error(excpt);
    }
    _dim = X.n_cols;
    arma::uvec P = arma::randperm(X.n_rows, _nfs);
    _data = X.rows(P);
    _stds = arma::stddev(_data,0);
    _data.each_row() /= _stds;
    X = _eval_kernel();
    if (use_arma_svd) arma::svd(_U, _s, _V, X);
    else random_sinks::powerSVD(_U, _s, _V, X, _nfs, max_iter, tol);
}

/* get_features(X) : represent X in the coordinate system of the principle components of the kernel matrix. */
arma::mat random_sinks::gauss_nystrom::get_features(arma::mat X) {
    if (X.n_cols != _dim) {
        std::string excpt = "get_features() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_dim) + ").\n";
        throw std::runtime_error(excpt);
    }
    X = _eval_kernel(X);
    X *= _V;
    return X;
}

/* _eval_kernel(X) : __private__ evaluate kernel for data X. */
arma::mat random_sinks::gauss_nystrom::_eval_kernel(arma::mat X) {
    X.each_row() /= _stds;
    arma::mat K(X.n_rows, _data.n_rows);
    for (uint i=0; i < X.n_rows; ++i) {
        for (uint j=0; j < _data.n_rows; ++j) {
            K(i,j) = std::exp(-_gamma*std::pow(arma::norm(X.row(i) - _data.row(j)),2));
        }
    }
    return K;
}

/* _eval_kernel() : __private__ evaluate Gram matrix.  */
arma::mat random_sinks::gauss_nystrom::_eval_kernel() {
    arma::mat K(_data.n_rows, _data.n_rows);
    K.diag().fill(0);
    for (uint i=0; i < _data.n_rows; ++i) {
        for (uint j=0; j < i; ++j) {
            K(i,j) = std::exp(-_gamma*std::pow(arma::norm(_data.row(i) - _data.row(j)),2));
        }
    }
    K = arma::symmatl(K);
    return K;
}