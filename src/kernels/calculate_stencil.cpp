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

    std::array<std::vector<multiindex>, 8> calculate_stencil() {
        std::array<std::vector<multiindex>, 8> stencils;

        // used to check the radiuses of the outer and inner sphere
        const real theta0 = opts.theta;

        // int64_t i0 = 0;
        // int64_t i1 = 0;
        // int64_t i2 = 0;

        for (int64_t i0 = 0; i0 < 2; ++i0) {
            for (int64_t i1 = 0; i1 < 2; ++i1) {
                for (int64_t i2 = 0; i2 < 2; ++i2) {
                    std::vector<multiindex> stencil;
                    for (int64_t j0 = i0 - INX; j0 < i0 + INX; ++j0) {
                        for (int64_t j1 = i1 - INX; j1 < i1 + INX; ++j1) {
                            for (int64_t j2 = i2 - INX; j2 < i2 + INX; ++j2) {
                                // don't interact with yourself!
                                if (i0 == j0 && i1 == j1 && i2 == j2) {
                                    continue;
                                }

				// indices on coarser level (for outer stencil boundary)
                                const int64_t i0_c = (i0 + INX) / 2 - INX / 2;
                                const int64_t i1_c = (i1 + INX) / 2 - INX / 2;
                                const int64_t i2_c = (i2 + INX) / 2 - INX / 2;

                                const int64_t j0_c = (j0 + INX) / 2 - INX / 2;
                                const int64_t j1_c = (j1 + INX) / 2 - INX / 2;
                                const int64_t j2_c = (j2 + INX) / 2 - INX / 2;

                                const real theta_f =
                                    detail::reciprocal_distance(i0, i1, i2, j0, j1, j2);
                                const real theta_c =
                                    detail::reciprocal_distance(i0_c, i1_c, i2_c, j0_c, j1_c, j2_c);

                                // if (j0 == 4 && j1 == 4 && j2 == -1) {
                                //     std::cout << "in 4,4,-1" << std::endl;
                                //     std::cout << "(i0, i1, i2) = (" << i0 << ", " << i1 << ", "
                                //     << i2 << ")"
                                //               << std::endl;
                                //     std::cout << "(j0, j1, j2) = (" << j0 << ", " << j1 << ", "
                                //     << j2 << ")"
                                //               << std::endl;
                                //     std::cout << "(i0_c, i1_c, i2_c) = (" << i0_c << ", " << i1_c
                                //     << ", "
                                //               << i2_c << ")" << std::endl;
                                //     std::cout << "(j0_c, j1_c, j2_c) = (" << j0_c << ", " << j1_c
                                //     << ", "
                                //               << j2_c << ")" << std::endl;
                                //     std::cout << "theta_f: " << theta_f << std::endl;
                                //     std::cout << "theta_c: " << theta_c << std::endl;
                                //     std::cout << "theta0: " << theta0 << std::endl;
                                // }

                                // not in inner sphere (theta_c > theta0), but in outer sphere
                                if (theta_c > theta0 && theta_f <= theta0) {
                                    stencil.emplace_back(j0 - i0, j1 - i1, j2 - i2);
                                }
                            }
                        }
                    }
                    stencils[i0 * 4 + i1 * 2 + i2] = stencil;
                }
            }
        }

        // for (size_t i = 0; i < 8; i++) {
        //     std::cout << "-------------- " << i << " ---------------" << std::endl;
        //     std::cout << "x, y, z" << std::endl;
        //     std::vector<multiindex>& stencil = stencils[i];
        //     for (multiindex& stencil_element : stencil) {
        //         std::cout << stencil_element << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        // std::vector<multiindex> common_stencil;
        // // use the first stencil to filter the elements that are in all other stencils
        // std::vector<multiindex>& first_stencil = stencils[0];
        // for (multiindex& first_element : first_stencil) {
        //     bool all_found = true;
        //     for (size_t i = 1; i < 8; i++) {
        //         bool found = false;
        //         for (multiindex& second_element : stencils[i]) {
        //             if (first_element.compare(second_element)) {
        //                 found = true;
        //                 break;
        //             }
        //         }
        //         if (!found) {
        //             all_found = false;
        //             break;
        //         }
        //     }
        //     if (all_found) {
        //         common_stencil.push_back(first_element);
        //     }
        // }

        // std::cout << "-------------- common_stencil"
        //           << " size: " << common_stencil.size() << " ---------------" << std::endl;
        // std::cout << "x, y, z" << std::endl;
        // for (multiindex& stencil_element : common_stencil) {
        //     std::cout << stencil_element << std::endl;
        // }
        // std::cout << std::endl;

        // std::array<std::vector<multiindex>, 8> diff_stencils;

        // for (size_t i = 0; i < 8; i++) {
        //     for (multiindex& element : stencils[i]) {
        //         bool found = false;
        //         for (multiindex& common_element : common_stencil) {
        //             if (element.compare(common_element)) {
        //                 found = true;
        //                 break;
        //             }
        //         }
        //         if (!found) {
        //             diff_stencils[i].push_back(element);
        //         }
        //     }
        // }

        // for (size_t i = 0; i < 8; i++) {
        //     std::cout << "-------------- diff_stencil: " << i
        //               << " size: " << diff_stencils[i].size() << " ---------------" << std::endl;
        //     std::cout << "x, y, z" << std::endl;
        //     std::vector<multiindex>& stencil = diff_stencils[i];
        //     for (multiindex& stencil_element : stencil) {
        //         std::cout << stencil_element << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

        return stencils;
    }

}    // namespace fmm
}    // namespace octotiger
