#include <random_kitchen_sinks.hpp>

bool is_pwr2(int k) {
    int e = std::floor(std::log2(k));
    return (std::exp2(e) == k);
}

/* fwht(X, apply_to_rows=true) : compute the Fast Walsh-Hadamard transform 
 * --- X : matrix to apply FWHT on. FWHT is computed on each row or each columns.
 * --- apply_to_rows : applies FWHT to each row if true, or to each column if false. */
arma::mat random_sinks::fwht(arma::mat X, bool apply_to_rows) {
    if (X.n_cols < 2 && X.n_rows < 2) {
        std::string excpt = "fwht() error: cannot apply Hadamard transform to matrix with less than 2 elements along both dimensions (X.size = " + std::to_string(X.n_rows) + " x " + std::to_string(X.n_cols) + ").\n";
        throw std::runtime_error(excpt);
    }
    if (X.n_rows == 1) apply_to_rows = true;
    if (X.n_cols == 1) apply_to_rows = false;
    if (apply_to_rows && !is_pwr2(X.n_rows)) {
        std::string excpt = "fwht() error: cannot apply transformation to rows because the number of columns is not a power of 2 (X.n_cols = " + std::to_string(X.n_cols) + ").\n";
        throw std::runtime_error(excpt);
    } else if (!apply_to_rows && !is_pwr2(X.n_cols)) {
        std::string excpt = "fwht() error: cannot apply transformation to columns because the number of rows is not a power of 2 (X.n_rows = " + std::to_string(X.n_rows) + ").\n";
        throw std::runtime_error(excpt);
    }

    if (apply_to_rows) {
        int h=1;
        arma::vec x,y;
        while (h < X.n_cols) {
            for (int i=0; i < X.n_cols; i+=2*h) {
                for (int j=i; j < i+h; ++j) {
                    x = X.col(j);
                    y = X.col(j+h);
                    X.col(j) = x + y;
                    X.col(j+h) = x - y;
                }
            }
            h*=2;
        }
    } else {
        int h=1;
        arma::rowvec x,y;
        while (h < X.n_rows) {
            for (int i=0; i < X.n_rows; i+=2*h) {
                for (int j=i; j < i+h; ++j) {
                    x = X.row(j);
                    y = X.row(j+h);
                    X.row(j) = x + y;
                    X.row(j+h) = x - y;
                }
            }
            h*=2;
        }
    }
    return X;
}