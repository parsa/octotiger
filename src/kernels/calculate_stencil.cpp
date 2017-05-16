#include "defs.hpp"
#include "geometry.hpp"
#include "options.hpp"

#include "calculate_stencil.hpp"

extern options opts;

namespace octotiger {
namespace fmm {
    namespace detail {

        // calculates 1/distance between i and j
        real reciprocal_distance(const integer i0, const integer i1, const integer i2,
            const integer j0, const integer j1, const integer j2) {
            real tmp = (sqr(i0 - j0) + sqr(i1 - j1) + sqr(i2 - j2));
            // protect against sqrt(0)
            if (tmp > 0.0) {
                return 1.0 / (std::sqrt(tmp));
            } else {
                return 1.0e+10;    // dummy value
            }
        }

        // checks whether the index tuple is inside the current node
        bool is_interior(const integer i0, const integer i1, const integer i2) {
            bool check = true;
            if (i0 < 0 || i0 >= G_NX) {
                check = false;
            } else if (i1 < 0 || i1 >= G_NX) {
                check = false;
            } else if (i2 < 0 || i2 >= G_NX) {
                check = false;
            }
            return check;
        }

        geo::direction get_neighbor_dir(const integer i, const integer j, const integer k) {
            integer i0 = 0, j0 = 0, k0 = 0;
            if (i < 0) {
                i0 = -1;
            } else if (i >= G_NX) {
                i0 = +1;
            }
            if (j < 0) {
                j0 = -1;
            } else if (j >= G_NX) {
                j0 = +1;
            }
            if (k < 0) {
                k0 = -1;
            } else if (k >= G_NX) {
                k0 = +1;
            }
            geo::direction d;
            d.set(i0, j0, k0);
            return d;
        }
    }

    std::vector<multiindex> calculate_stencil() {
        std::vector<multiindex> stencil;

        // used to check the radiuses of the outer and inner sphere
        const real theta0 = opts.theta;

        int64_t i0 = 0;
        int64_t i1 = 0;
        int64_t i2 = 0;

        for (int64_t j0 = i0 - INX; j0 < i0 + INX; ++j0) {
            for (int64_t j1 = i1 - INX; j1 < i1 + INX; ++j1) {
                for (int64_t j2 = i2 - INX; j2 < i2 + INX; ++j2) {
                    // don't interact with yourself!
                    if (i0 == j0 && i1 == j1 && i2 == j2) {
                        continue;
                    }

                    const int64_t i0_c = (i0 + INX) / 2 - INX / 2;
                    const int64_t i1_c = (i1 + INX) / 2 - INX / 2;
                    const int64_t i2_c = (i2 + INX) / 2 - INX / 2;

                    const int64_t j0_c = (j0 + INX) / 2 - INX / 2;
                    const int64_t j1_c = (j1 + INX) / 2 - INX / 2;
                    const int64_t j2_c = (j2 + INX) / 2 - INX / 2;

                    const real theta_f = detail::reciprocal_distance(i0, i1, i2, j0, j1, j2);
                    const real theta_c =
                        detail::reciprocal_distance(i0_c, i1_c, i2_c, j0_c, j1_c, j2_c);

                    // not in inner sphere (theta_c > theta0), but in outer sphere
                    if (theta_c > theta0 && theta_f <= theta0) {
                        stencil.emplace_back(j0, j1, j2);
                    }
                }
            }
        }
        return stencil;
    }

}    // namespace fmm
}    // namespace octotiger
