#pragma once

#include <armadillo>

namespace random_sinks {
    class RFF {
        private:
        arma::mat _w;
        arma::rowvec _b;
        double _gamma;
        int _nfs, _dim;

        public:
        const arma::mat& weights;
        const arma::rowvec& bias;

        RFF(double gamma=1, int n_feats=100, int seed=-1);
        void fit(const arma::mat& X);
        arma::mat get_features(const arma::mat& X);
    };

    arma::mat fwht(arma::mat X, bool apply_to_rows=true);

    inline arma::vec randchi(double df, uint n) {
        return arma::sqrt(arma::chi2rnd(df, n));
    }

    class fastfood {
        private:
        arma::vec _B, _G, _S, _b;
        arma::uvec _P;
        int _dim, _nfs, _d;
        double _gamma;

        public:
        fastfood(double gamma=1, int n_feats=128, int seed=-1);
        void fit(const arma::mat& X);
        arma::mat get_features(arma::mat X);
    };
}