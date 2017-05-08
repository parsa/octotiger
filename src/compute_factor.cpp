#include "taylor.hpp"

extern taylor<4, real> factor;

void compute_factor() {
    factor = 0.0;
    factor() += 1.0;
    for (integer a = 0; a < NDIM; ++a) {
        factor(a) += 1.0;
        for (integer b = 0; b < NDIM; ++b) {
            factor(a, b) += 1.0;
            for (integer c = 0; c < NDIM; ++c) {
                factor(a, b, c) += 1.0;
            }
        }
    }
}
