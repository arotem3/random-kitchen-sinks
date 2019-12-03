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

    class SORF {
        private:
        arma::rowvec _B1, _B2, _B3, _b;
        int _dim, _d, _nfs;
        double _gamma;

        public:
        SORF(double gamma=1, int n_feats=128, int seed=-1);
        void fit(const arma::mat& X);
        arma::mat get_features(arma::mat X);
    };

    void rSVD(arma::mat& U, arma::vec& s, arma::mat& V, const arma::mat& X, uint n_comps);
    double powerSVD(arma::mat& U, arma::vec& s, arma::mat& V, const arma::mat& X, uint n_comps, uint max_iter=10, double tol=1e-8);

    class tPCA {
        private:
        arma::mat _V;
        arma::vec _s;
        arma::rowvec _means;
        int _dim, _n_comps;
        bool _use_rSVD;

        public:
        const arma::mat& components;
        const arma::vec& cov_eigenvals;
        const int& n_components;

        tPCA(uint n_components, std::string method="rSVD");
        void fit(arma::mat X, uint max_iter=10, double tol=1e-8, bool zero_mean=false);
        arma::mat get_features(arma::mat X);
        arma::mat get_projection(arma::mat X);
        arma::mat inv_features(arma::mat X);
    };

    class gauss_nystrom {
        private:
        arma::mat _data, _V, _U;
        arma::vec _s;
        arma::rowvec  _stds;
        double _gamma;
        int _dim, _nfs;
        arma::mat _eval_kernel(arma::mat X);
        arma::mat _eval_kernel();

        public:
        const arma::vec& kernel_singular_vals;
        const arma::mat& kernel_left_vecs;
        const arma::mat& kernel_right_vecs;
        gauss_nystrom(uint n_feats=128, double gamma=1, int seed=-1);
        void fit(arma::mat X, std::string method="powerSVD", uint max_iter=10, double tol=1e-8);
        arma::mat get_features(arma::mat X);
    };
}