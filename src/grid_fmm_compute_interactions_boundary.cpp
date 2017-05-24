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
    auto start = std::chrono::high_resolution_clock::now();
    auto& M = *M_ptr;
    // auto& mon = *mon_ptr;

    // always reinitialized in innermost loop
    std::array<simd_vector, NDIM> dX;
    taylor<5, simd_vector> D;
    taylor<4, simd_vector> n0;

    std::vector<space_vector> const& com0 = *(com_ptr[0]);
    size_t list_size = ilist_n_bnd.size();
    for (size_t si = 0; si < list_size; si++) {
        std::array<simd_vector, NDIM> X;
        boundary_interaction_type const& bnd = ilist_n_bnd[si];

        for (integer i = 0; i < simd_len; ++i) {
            // const integer iii0 = bnd.first[0 + i];
            const integer iii0 = bnd.first[0];
            space_vector const& com0iii0 = com0[iii0];
            for (integer d = 0; d < NDIM; ++d) {
                X[d] = com0iii0[d];
            }
        }

        taylor<4, simd_vector> A0;
        std::array<simd_vector, NDIM> B0;

        taylor<4, simd_vector> m1;
        if (type == RHO) {
            const integer iii0 = bnd.first[0];

            multipole const& Miii0 = M[iii0];
            m1[0] = Miii0[0];
            for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                m1[j] = Miii0[j];
            }
        }

        const integer list_size = bnd.second.size();
        // TODO: is that always divisible? -> padding?
        for (integer li = 0; li < list_size; li += simd_len) {
            taylor<4, simd_vector> m0;
            std::array<simd_vector, NDIM> Y;

            for (integer i = 0; i < simd_len && li + i < list_size; ++i) {
                integer index = mpoles.is_local ? bnd.second[li + i] :
                                                  li + i;    // TODO: can this line be removed?
                auto const& tmp1 = (*(mpoles.M))[index];
                for (int j = 0; j != 20; ++j) {
                    m0[j][i] = tmp1[j];
                }

                auto const& tmp2 = (*(mpoles.x))[index];
                for (integer d = 0; d != NDIM; ++d) {
                    Y[d][i] = tmp2[d];
                }
            }

            if (type == RHO) {
                simd_vector tmp = m0[0] / m1[0];
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = m0[j] - m1[j] * tmp;
                }
            } else {
                // reset used range
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = 0.0;
                }
            }

            for (integer d = 0; d < NDIM; ++d) {
                dX[d] = X[d] - Y[d];
            }

            D.set_basis(dX);
            A0[0] += m0[0] * D[0];
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

                A0(a) += m0() * D(a);
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
                A0[ab_idx] += m0() * D[ab_idx];
                for (integer c = 0; c < NDIM; ++c) {
                    A0[ab_idx] -= m0(c) * D[abc_idx_map6[c]];
                }
            }

            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                A0[i] += m0[0] * D[i];
            }

            // // print interaction information for debugging
            // for (integer i = 0; i != simd_len; ++i) {
            //     // if (interaction_list[interaction_first_index + i].first == 0 &&
            //     //     interaction_list[interaction_second_index + i].second == 3) {
            //     auto& first_index = bnd.first_index[0];
            //     if (first_index[0] == 0 && first_index[1] == 0 && first_index[2] == 0) {
            //         std::cout << "---------------------- boundary ---------------------------"
            //                   << std::endl;

            //         std::cout << "first_index: " << first_index[0] << ", " << first_index[1] <<
            //         ", "
            //                   << first_index[2];
            //         auto& second_index = bnd.second_index[li + i];
            //         std::cout << " second_index: " << second_index[0] << ", " << second_index[1]
            //                   << ", " << second_index[2];
            //         std::cout << " diff: " << (second_index[0] - first_index[0]) << ", "
            //                   << (second_index[1] - first_index[1]) << ", "
            //                   << (second_index[2] - first_index[2]) << std::endl;
            //         std::cout << "first_index_flat: " << bnd.first[0] << std::endl;
            //         std::cout << "second_index_flat: " << bnd.second[li + i] << std::endl;
            //         std::cout << "X: " << X[0][i] << ", " << X[1][i] << ", " << X[2][i]
            //                   << std::endl;
            //         std::cout << "Y: " << Y[0][i] << ", " << Y[1][i] << ", " << Y[2][i]
            //                   << std::endl;
            //         std::cout << "m0 (m_partner): ";
            //         for (size_t j = 0; j < m0.size(); j++) {
            //             if (j > 0) {
            //                 std::cout << ", ";
            //             }
            //             std::cout << m0[j][i];
            //         }
            //         std::cout << std::endl;
            //         std::cout << "D: ";
            //         for (size_t j = 0; j < D.size(); j++) {
            //             if (j > 0) {
            //                 std::cout << ", ";
            //             }
            //             std::cout << D[j][i];
            //         }
            //         std::cout << std::endl;
            //         std::cout << "A0: ";
            //         for (size_t j = 0; j < A0.size(); j++) {
            //             if (j > 0) {
            //                 std::cout << ", ";
            //             }
            //             std::cout << A0[j][i];
            //         }
            //         std::cout << std::endl;

            //         std::cout << "B0: ";
            //         for (size_t d = 0; d < NDIM; d++) {
            //             if (d > 0) {
            //                 std::cout << ", ";
            //             }
            //             std::cout << B0[d][i];
            //         }
            //         std::cout << std::endl;
            //     }
            // }
        }

        const integer iii0 = bnd.first[0];
        expansion& Liii0 = L[iii0];

        for (integer j = 0; j != taylor_sizes[3]; ++j) {
#if Vc_IS_VERSION_2 == 0
            Liii0[j] += A0[j].sum();
#else
            Liii0[j] += Vc::reduce(A0[j]);
#endif
        }

        if (type == RHO && opts.ang_con) {
            space_vector& L_ciii0 = L_c[iii0];
            for (integer j = 0; j != NDIM; ++j) {
#if Vc_IS_VERSION_2 == 0
                L_ciii0[j] += B0[j].sum();
#else
                L_ciii0[j] += Vc::reduce(B0[j]);
#endif
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "boundary compute_interactions_inner duration (ms, branch): " << duration.count()
              << std::endl;
    PROF_END;
}
