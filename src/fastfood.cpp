#include <random_kitchen_sinks.hpp>

/* fastfood(gamma=1, n_feats=128, int seed=-1) : initialize fastfood object.
 * --- gamma : parameter of Gaussian kernel, i.e. exp(-gamma * ||x||^2)
 * --- n_feats : number of random features, must be a power of two, if not, then it will be rounded up to a power of two.
 * --- seed : random seed, if seed<0 then the seed will be randomized. */
random_sinks::fastfood::fastfood(double gamma, int n_feats, int seed) {
    if (gamma <= 0 || n_feats <= 1) {
        std::string excpt = "fastfood() error: invalid parameter values.\n\tgamma = " + std::to_string(gamma) + " (must be > 0)\n\tn_feats = " + std::to_string(n_feats) + " (must be > 1...much greater...)\n";
        throw std::runtime_error(excpt);
    }
    _nfs = std::exp2(std::ceil(std::log2(n_feats)));
    _gamma = gamma;
    if (seed < 0) arma::arma_rng::set_seed_random();
    else arma::arma_rng::set_seed(seed);
}

/* fit(x) : produce the various coefficient vectors.
 * --- x : matrix whose dimensions we match. */
void random_sinks::fastfood::fit(const arma::mat& X) {
    _dim = X.n_cols;
    _d = std::exp2(std::ceil(std::log2(_nfs)));
    _B = 2*arma::randi<arma::rowvec>(_nfs, arma::distr_param(0,1)) - 1;
    _P = arma::randperm(_nfs);
    _G = arma::randn<arma::rowvec>(_nfs);
    _S = 2*_gamma / (std::sqrt(_d)*arma::norm(_G,2)) * random_sinks::randchi<arma::rowvec>(_d, _nfs);
    _b = arma::randu<arma::rowvec>(_nfs) * (2*M_PI);
}

/* get_features(x) : evaluate random features.
 * --- x : matrix to evaluate features for. */
arma::mat random_sinks::fastfood::get_features(arma::mat X) {
    if (X.n_cols != _dim) {
        std::string excpt = "fastfood::get_features() error: dimension mismatch. X.n_cols = " + std::to_string(X.n_cols) + " (must be " + std::to_string(_dim) + ").\n";
        throw std::runtime_error(excpt);
    }

    if (_dim != _d) { // zero-fill
        X.reshape(X.n_rows, _d);
    }

    arma::mat WX = arma::zeros(X.n_rows, _nfs);
    arma::mat wx;
    if (_d > _nfs) {
        for (int i=0; i < _d; i+=_nfs) {
            wx = X.cols(i,i+_nfs-1).each_row() % _B;
            wx = fwht(wx);
            wx = wx.cols(_P);
            wx = fwht(wx.each_row() % _G);
            wx.each_row() %= _S;
            WX += wx;
        }
    } else {
        for (int i=0; i < _nfs; i+=_d) {
            WX.cols(i, i+_d-1) = fwht(X.each_row() % _B.cols(i,i+_d-1));
        }
        WX = WX.cols(_P);
        WX.each_row() %= _G;
        for (int i=0; i < _nfs; i+=_d) {
            WX.cols(i, i+_d-1) = fwht(WX.cols(i,i+_d-1));
        }
        WX.each_row() %= _S;
    }
    WX = arma::cos(WX.each_row() + _b);
    return WX;
}