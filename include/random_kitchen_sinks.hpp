#pragma once

#include <armadillo>

namespace random_sinks {
    arma::mat fwht(arma::mat X, bool apply_to_rows=true);

    template<class T> inline T randchi(double df, uint m, uint n) { // T must be an armadillo Row, Col, or Mat type
        return arma::sqrt(arma::chi2rnd<T>(df, m, n));
    }

    template<class T> inline T randchi(double df, uint n) { // T must be an armadillow Row or Col type
        return arma::sqrt(arma::chi2rnd<T>(df, n));
    }

    inline double randchi(double df) {
        return std::sqrt(arma::chi2rnd<double>(df));
    }
    
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

    class fastfood {
        private:
        arma::rowvec _B, _G, _S, _b;
        arma::uvec _P;
        int _dim, _nfs, _d;
        double _gamma;

        public:
        fastfood(double gamma=1, int n_feats=128, int seed=-1);
        void fit(const arma::mat& X);
        arma::mat get_features(arma::mat X);
    };
}