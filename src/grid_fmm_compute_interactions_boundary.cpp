#include "grid.hpp"
#include "grid_flattened_indices.hpp"
#include "options.hpp"
#include "physcon.hpp"
#include "profiler.hpp"
#include "simd.hpp"
#include "taylor.hpp"

#include <hpx/include/parallel_for_loop.hpp>

#include <chrono>
#include <cstddef>
#include <utility>

extern taylor<4, real> factor;
extern options opts;

void grid::compute_boundary_interactions_multipole_multipole(gsolve_type type,
    const std::vector<boundary_interaction_type>& ilist_n_bnd,
    const gravity_boundary_type& mpoles) {
    PROF_BEGIN;
    // auto start = std::chrono::high_resolution_clock::now();
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

    std::vector<space_vector> const& com0 = *(com_ptr[0]);
    size_t list_size = ilist_n_bnd.size();
    for (size_t si = 0; si < list_size; si++) {
        taylor<4, simd_vector> m0;
        taylor<4, simd_vector> n0;
        std::array<simd_vector, NDIM> dX;
        std::array<simd_vector, NDIM> X;
        std::array<simd_vector, NDIM> Y;
        // space_vector Y;

        boundary_interaction_type const& bnd = ilist_n_bnd[si];

        // TODO: remove after changes
        if (bnd.first.size() == 0) {
            continue;
        }

        // TODO: remove after changes
        if (bnd.second.size() == 0) {
            continue;
        }

        for (integer i = 0; i < simd_len; ++i) {
            // const integer iii0 = bnd.first[0 + i];
            const integer iii0 = bnd.first[0];
            space_vector const& com0iii0 = com0[iii0];
            for (integer d = 0; d < NDIM; ++d) {
                X[d] = com0iii0[d];
            }
        }

        const integer list_size = bnd.second.size();    // TODO: changed as well!
        for (integer li = 0; li < list_size; li += simd_len) {
            // for (integer li = 0; li < list_size; li += 1) {

            // for (integer i = 0; i < 1; ++i) {
            for (integer i = 0; i < simd_len && li + i < list_size; ++i) {
                integer index = mpoles.is_local ? bnd.second[li + i] :
                                                  li + i;    // TODO: can this line be removed?
                auto const& tmp1 = (*(mpoles.M))[index];
#pragma GCC ivdep
                for (int j = 0; j != 20; ++j) {
                    m0[j][i] = tmp1[j];
                }
                auto const& tmp2 = (*(mpoles.x))[index];
#pragma GCC ivdep
                for (integer d = 0; d != NDIM; ++d) {
                    Y[d][i] = tmp2[d];
                }
            }

            // std::array<simd_vector, NDIM> simdY = {
            //     simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])),
            // };

            if (type == RHO) {
                // for (integer i = 0; i < 1; ++i) {
                for (integer i = 0; i < simd_len && li + i < list_size;
                     ++i) {    // TODO: very convoluted, in all lanes the same?
                    // for (integer i = 0; i < simd_len; ++i) { //TODO: very convoluted, in all
                    // lanes the same?
                    // const integer iii0 = bnd.first[0 + i];
                    const integer iii0 = bnd.first[0];

                    multipole const& Miii0 = M[iii0];
                    real const tmp = m0()[i] / Miii0();
                    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                        n0[j][i] = m0[j][i] - Miii0[j] * tmp;
                    }
                }
            }

            if (type != RHO) {
#pragma GCC ivdep
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = ZERO;
                }
            }
#pragma GCC ivdep
            for (integer d = 0; d < NDIM; ++d) {
                // dX[d] = X[d] - simdY[d];
                dX[d] = X[d] - Y[d];
            }

            taylor<5, simd_vector> D;
            taylor<4, simd_vector> A0;
            std::array<simd_vector, NDIM> B0 = {
                simd_vector(0.0), simd_vector(0.0), simd_vector(0.0)};

            D.set_basis(dX);
            A0[0] = m0[0] * D[0];
            if (type != RHO) {
                for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
                    A0[0] -= m0[i] * D[i];
                }
            }
            for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
                A0[0] += m0[i] * D[i] * (factor[i] * HALF);
            }
            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                A0[0] -= m0[i] * D[i] * (factor[i] * SIXTH);
            }

            for (integer a = 0; a < NDIM; ++a) {
                int const* ab_idx_map = to_ab_idx_map3[a];
                int const* abc_idx_map = to_abc_idx_map3[a];

                A0(a) = m0() * D(a);
                for (integer i = 0; i != 6; ++i) {
                    if (type != RHO && i < 3) {
                        A0(a) -= m0(a) * D[ab_idx_map[i]];
                    }
                    const integer cb_idx = cb_idx_map[i];
                    const auto tmp = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                    A0(a) += m0[cb_idx] * tmp;
                }
            }

            if (type == RHO) {
                for (integer a = 0; a < NDIM; ++a) {
                    int const* abcd_idx_map = to_abcd_idx_map3[a];
                    for (integer i = 0; i != 10; ++i) {
                        const integer bcd_idx = bcd_idx_map[i];
                        const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                        B0[a] -= n0[bcd_idx] * tmp;
                    }
                }
            }

            for (integer i = 0; i != 6; ++i) {
                int const* abc_idx_map6 = to_abc_idx_map6[i];

                integer const ab_idx = ab_idx_map6[i];
                A0[ab_idx] = m0() * D[ab_idx];
                for (integer c = 0; c < NDIM; ++c) {
                    A0[ab_idx] -= m0(c) * D[abc_idx_map6[c]];
                }
            }

            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                A0[i] = m0[0] * D[i];
            }

            for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
                // for (integer i = 0; i < 1; ++i) {
                // const integer iii0 = bnd.first[li + i];
                const integer iii0 = bnd.first[0];
                expansion& Liii0 = L[iii0];

#pragma GCC ivdep
                for (integer j = 0; j != taylor_sizes[3]; ++j) {
                    Liii0[j] += A0[j][i];
                }
            }

            if (type == RHO && opts.ang_con) {
                for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
                    // for (integer i = 0; i < 1; ++i) {
                    // const integer iii0 = bnd.first[li + i];
                    const integer iii0 = bnd.first[0];
                    space_vector& L_ciii0 = L_c[iii0];
#pragma GCC ivdep
                    for (integer j = 0; j != NDIM; ++j) {
                        L_ciii0[j] += B0[j][i];
                    }
                }
            }
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << "boundary compute_interactions_inner duration (ms): " << duration.count() <<
    // std::endl;
    PROF_END;
}
