#include<random_kitchen_sinks.hpp>

/* gauss_nystrom(n_components, gamma=1, method="rSVD") : initialize gauss nystrom kernel pca object.
 * --- n_components : number of principle components to compute.
 * --- method : method to compute PCA with, either "rSVD" or "powerSVD" */
random_sinks::gauss_nystrom::gauss_nystrom(uint n_components, double gamma, std::string method) : pca(n_components, method) {
    if (gamma <= 0) {
        std::string excpt = "gauss_nystrom() error: gamma parameter must be strictly positive, gamma requested = " + std::to_string(gamma) + "\n";
        throw std::logic_error(excpt);
    }
    _gamma = gamma;
}

/* fit(X, max_iter=10, tol=1e-8) : fit kernel pca.
 * --- X : data matrix.
 * --- max_iter, tol : if powerSVD is being used, these parameters are passed along to it, otherwise they are ignored. */
void random_sinks::gauss_nystrom::fit(arma::mat X, uint max_iter, double tol) {
    _stds = arma::stddev(X,0);
    _data = X; _data.each_row() /= _stds;
    _dim = X.n_cols;
    X = _eval_kernel();
    pca.fit(X, max_iter, tol);
}

/* get_features(X) : represent X in the coordinate system of the principle components of the kernel matrix. */
arma::mat random_sinks::gauss_nystrom::get_features(arma::mat X) {
    if (X.n_cols != _dim) {
        std::string excpt = "get_features() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_dim) + ").\n";
        throw std::logic_error(excpt);
    }
    X = _eval_kernel(X);
    X = pca.get_features(X);
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