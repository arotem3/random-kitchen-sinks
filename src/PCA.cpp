#include<random_kitchen_sinks.hpp>

/* powerSVD(U, s, V, X, n_comps, max_iter=10, tol=1e-8) : computed the n_comp truncated SVD of a matrix X using power iteration.
 * --- U, s, V : are such that X ~ U * diagmat(s) * V.t()
 * --- X : matrix to approximate SVD for.
 * --- n_comps : number of singular values and respective vectors to compute.
 * --- max_iter : maximum number of power iterations, by default this is 1, and should be sufficient to get a very small error in most cases.
 * --- tol : tolerance such that if the error in the singular values is less than tol, then iteration will stop before max_iter is reached. */
double random_sinks::powerSVD(arma::mat& U, arma::vec& s, arma::mat& V, const arma::mat& X, uint n_comps, uint max_iter, double tol) {
    if (n_comps == 0) {
        std::string s = "powerSVD() error: n_components must be greater than zero.\n";
        throw std::logic_error(s);
    } else if (n_comps > X.n_cols) {
        std::string s = "powerSVD() error: number of components requested (=" + std::to_string(n_comps) + ") exceeds number of columns in X (X.n_cols=" + std::to_string(X.n_cols) + ")\n";
        throw std::logic_error(s);
    } else if (n_comps > 0.75*X.n_cols) {
        std::cerr << "powerSVD() warning: number of components requested (=" << n_comps << ") is relatively large (X.n_cols=" << X.n_cols << ") it may be more efficient to compute the full SVD instead.\n";
    }

    double err;
    arma::mat R; // dummy variable for QR
    s = arma::zeros(n_comps);

    if (X.n_cols <= X.n_rows) { // user power iteration to compute eigen-decomp of X'*X
        arma::mat V1 = arma::randn(X.n_cols, n_comps);
        arma::qr_econ(V, R, V1);
        for (uint i=0; i < max_iter; ++i) {
            V1 = X.t() * X * V;
            for (uint j=0; j < n_comps; ++j) {
                s(j) = arma::dot(V.col(j), V1.col(j));
            }
            arma::qr_econ(V, R, V1);
            err = arma::norm(V.each_row()%s.t() - V1, "inf");
            if (err < tol) break;
        }
        s = arma::sqrt(s);
        U = X * V; U.each_row() /= s.t();
    } else { // use power iteration to compute eigen-decomp of X*X'
        arma::mat U1 = arma::randn(X.n_cols, n_comps);
        arma::qr_econ(U, R, U1);
        for (uint i=0; i < max_iter; ++i) {
            U1 = X * X.t() * U;
            for (uint j=0; j < n_comps; ++j) {
                s(j) = arma::dot(U.col(j), U1.col(j));
            }
            arma::qr_econ(U, R, U1);
            err = arma::norm(U.each_row()%s.t() - U1, "inf");
            if (err < tol) break;
        }
        s = arma::sqrt(s);
        V = X.t() * U; V.each_row() /= s.t();
    }
    
    return err;
}

/* rSVD(U, s, V, X, n_comps) : computed the n_comp truncated SVD of a matrix X using the fast randomized range finder algorithm.
 * --- U, s, V : are such that X ~ U * diagmat(s) *  V.t()
 * --- X : matrix to approximate SVD for.
 * --- n_comps : number of singular values and respective vectors to compute. */
void random_sinks::rSVD(arma::mat& U, arma::vec& s, arma::mat& V, const arma::mat& X, uint n_comps) {
    if (n_comps == 0) {
        std::string s = "rSVD() error: n_components must be greater than zero.\n";
        throw std::logic_error(s);
    } else if (n_comps > X.n_cols) {
        std::string s = "rpwrSVD() error: number of components requested (=" + std::to_string(n_comps) + ") exceeds number of columns in X (X.n_cols=" + std::to_string(X.n_cols) + ")\n";
        throw std::logic_error(s);
    } else if (n_comps > 0.75*X.n_cols) {
        std::cerr << "rpwrSVD() warning: number of components requested (=" << n_comps << ") is relatively large (X.n_cols=" << X.n_cols << ") it may be more efficient to compute the full SVD instead.\n";
    }
    int m = X.n_cols;

    uint nc2 = std::exp2(std::ceil(std::log2(n_comps)));
    uint m2 = std::exp2(std::ceil(std::log2(m)));

    arma::rowvec omega = 2*arma::randi<arma::rowvec>(nc2, arma::distr_param(0,1)) - 1;
    arma::uvec P = arma::randperm(m, nc2);
    
    arma::mat XO = X.cols(P); XO.each_row() %= omega;
    XO = random_sinks::fwht(XO, true);
    XO = XO.cols(0,n_comps-1);

    arma::mat Q;
    arma::qr_econ(Q, V, XO); // V is used temporarily as a placeholder

    arma::mat B = Q.t() * X;
    arma::svd_econ(U, s, V, B);
    U = Q * U;
}

/* tPCA(n_components, method="rSVD") : initialize truncated PCA object.
 * --- n_components : number of principle components to compute.
 * --- method : method to compute PCA with, either "rSVD" or "powerSVD" */
random_sinks::tPCA::tPCA(uint n_components, std::string method) : components(_V), cov_eigenvals(_s) {
    if (method == "rSVD") _use_rSVD = true;
    else if (method == "powerSVD") _use_rSVD = false;
    else {
        std::string s = "tPCA() error: method must be one of {\"rSVD\", \"powerSVD\"}. Method requested was \"" + method + "\".\n";
        throw std::logic_error(s);
    }
    _n_comps = n_components;
}

/* fit(X, max_iter=10, tol=1e-8, zero_mean=false) : fit PCA object.
 * --- X : data matrix.
 * --- max_iter, tol : if powerSVD is being used, these parameters are passed along to it, otherwise they are ignored.
 * --- zero_mean : if set to true, the object will not compute the column means, otherwise the column means will be computed. Typically, the column means are subtracted away in PCA. */
void random_sinks::tPCA::fit(arma::mat X, uint max_iter, double tol, bool zero_mean) {
    if (!zero_mean) {
        _means = arma::mean(X, 0);
    }
    X.each_row() -= _means;
    _dim = X.n_cols;
    arma::mat U;
    if (_use_rSVD) random_sinks::rSVD(U, _s, _V, X, _n_comps);
    else random_sinks::powerSVD(U, _s, _V, X, _n_comps, max_iter, tol);
    _s = arma::square(_s);
}

/* get_features(X) : represent X in the coordinate system of the principle components. */
arma::mat random_sinks::tPCA::get_features(arma::mat X) {
    if (X.n_cols != _dim) {
        std::string excpt = "tPCA::get_features() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_dim) + ").\n";
        throw std::runtime_error(excpt);
    }
    if (!_means.is_empty()) {
        X.each_row() -= _means;
    }
    X = X * _V;
    return X;
}

/* inv_features(X) : project PCA components back into original space.  */
arma::mat random_sinks::tPCA::inv_features(arma::mat X) {
    if (X.n_cols != _n_comps) {
        std::string excpt = "tPCA::inv_features() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_n_comps) + ").\n";
        throw std::runtime_error(excpt);
    }
    X = X * _V.t();
    if (!_means.is_empty()) {
        X.each_row() += _means;
    }
    return X;
}

/* get_projection(X) : represent X in the original space, projected onto the principle components subspace. */
arma::mat random_sinks::tPCA::get_projection(arma::mat X) {
    if (X.n_cols != _dim) {
        std::string excpt = "tPCA::get_projection() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_dim) + ").\n";
        throw std::runtime_error(excpt);
    }
    if (!_means.is_empty()) {
        X.each_row() -= _means;
    }
    X = X * _V * _V.t();
    if (!_means.is_empty()) {
        X.each_row() += _means;
    }
    return X;
}