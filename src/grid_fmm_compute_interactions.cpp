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

// non-root and non-leaf node (David)
extern std::vector<interaction_type> ilist_n;
// monopole-monopole interactions? (David)
extern std::vector<interaction_type> ilist_d;
// interactions of root node (empty list? boundary stuff?) (David)
extern std::vector<interaction_type> ilist_r;
extern std::vector<std::vector<boundary_interaction_type>> ilist_d_bnd;
extern std::vector<std::vector<boundary_interaction_type>> ilist_n_bnd;
extern taylor<4, real> factor;
extern options opts;

void grid::compute_interactions(gsolve_type type) {
    PROF_BEGIN;

    if (!is_leaf) {
        compute_interactions_inner(type);
    } else {
        compute_interactions_leaf(type);
    }

    PROF_END;
}

void grid::compute_interactions_inner(gsolve_type type) {
    auto start = std::chrono::high_resolution_clock::now();

    // calculating the contribution of all the inner cells
    // calculating the interaction

    auto& mon = *mon_ptr;
    // L stores the gravitational potential
    // L should be L, (10) in the paper
    // L_c stores the correction for angular momentum
    // L_c, (20) in the paper (Dominic)
    std::fill(std::begin(L), std::end(L), ZERO);
    if (opts.ang_con) {
        std::fill(std::begin(L_c), std::end(L_c), ZERO);
    }

    // Non-leaf nodes use taylor expansion
    // For leaf node, calculates the gravitational potential between two particles

    // Non-leaf means that multipole-multipole interaction (David)
    // Fetch the interaction list for the current cell (David)
    auto& interaction_list = is_root ? ilist_r : ilist_n;
    const integer list_size = interaction_list.size();

    // Space vector is a vector pack (David)
    // Center of masses of each cell (com_ptr1 is the center of mass of the parent cell)
    std::vector<space_vector> const& com0 = *(com_ptr[0]);
    // Do 8 cell-cell interactions at the same time (simd.hpp) (Dominic)
    // This could lead to problems either on Haswell or on KNL as the number is hardcoded
    // (David)
    // It is unclear what the underlying vectorization library does, if simd_len > architectural
    // length (David)
    // hpx::parallel::for_loop_strided(for_loop_policy, 0, list_size, simd_len,
    //     [&com0, &interaction_list, list_size, type, this](std::size_t li) {

    // for (size_t li = 0; li < list_size; li += 1) {
    //   std::cout << "li: " << li << " first: " << interaction_list[li].first << " second: " <<
    // interaction_list[li].second << " inner_loop_stop: " << interaction_list[li].inner_loop_stop
    // << std::endl;
    // }

    // std::cout << "init lasts to 0...: " << std::endl;
    // taylor<4, simd_vector> A0_last, A1_last;

    // std::cout << "list_size:" << list_size << std::endl;

    size_t interaction_first_index = 0;
    while (interaction_first_index < list_size) {    // simd_len
        // std::cout << "interaction_first_index: " << interaction_first_index << std::endl;

        std::array<simd_vector, NDIM> X;
        // taylor<4, simd_vector> m1;

        auto& M = *M_ptr;

        // TODO: only uses first 10 coefficients? ask Dominic
        // TODO: ask Dominic about padding
        // load variables from the same first multipole into all vector registers
        const integer iii0 = interaction_list[interaction_first_index].first;
        space_vector const& com0iii0 = com0[iii0];
        for (integer d = 0; d < NDIM; ++d) {
            // load the 3D center of masses
            X[d] = com0iii0[d];
        }

        // cell specific taylor series coefficients
        // multipole const& Miii0 = M[iii0];
        // for (integer j = 0; j != taylor_sizes[3]; ++j) {
        //     m1[j] = Miii0[j];
        // }

        taylor<4, simd_vector> A0;
        std::array<simd_vector, NDIM> B0 = {
            simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO)};

        // stop index of the iteration -> points to first entry in interaction_list where the first
        // multipole doesn't match the current multipole
        size_t inner_loop_stop = interaction_list[interaction_first_index].inner_loop_stop;

        // to a simd-sized step in the sublist of the interaction list where the outer multipole is
        // same
        // TODO: there is a possible out-of-bounds here, needs padding
        // for (size_t interaction_second_index = interaction_first_index; interaction_second_index
        // < inner_loop_stop; interaction_second_index += simd_len) {
        for (size_t interaction_second_index = interaction_first_index;
             interaction_second_index < inner_loop_stop; interaction_second_index += simd_len) {
            // std::cout << "    -> interaction_second_index: " << interaction_second_index <<
            // std::endl;

            std::array<simd_vector, NDIM> Y;
            taylor<4, simd_vector> m0;

            // TODO: this is the one remaining expensive gather step, needed Bryces gather approach
            // here
            // load variables for the second multipoles, one in each vector lane
            for (integer i = 0;
                 i != simd_len && integer(interaction_second_index + i) < inner_loop_stop; ++i) {
                const integer iii1 = interaction_list[interaction_second_index + i].second;
                space_vector const& com0iii1 = com0[iii1];
                for (integer d = 0; d < NDIM; ++d) {
                    // load the 3D center of masses
                    Y[d][i] = com0iii1[d];
                }

                // cell specific taylor series coefficients
                multipole const& Miii1 = M[iii1];
                for (integer j = 0; j != taylor_sizes[3]; ++j) {
                    m0[j][i] = Miii1[j];
                }
            }

            // taylor<4, simd_vector> n1;
            // n angular momentum of the cells
            taylor<4, simd_vector> n0;

            // Initalize moments and momentum
            for (integer i = 0;
                 i != simd_len && integer(interaction_second_index + i) < inner_loop_stop; ++i) {
                if (type != RHO) {
// TODO: Can we avoid that?
#pragma GCC ivdep
                    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                        n0[j] = ZERO;
                        // n1[j] = ZERO;
                    }
                } else {
                    // TODO: let's hope this is enter rarely
                    const integer iii0 = interaction_list[interaction_first_index].first;
                    const integer iii1 = interaction_list[interaction_second_index + i].second;
                    multipole const& Miii0 = M[iii0];
                    multipole const& Miii1 = M[iii1];
                    // this branch computes the angular momentum correction, (20) in the paper
                    // divide by mass of other cell
                    real const tmp1 = Miii1() / Miii0();
                    real const tmp2 = Miii0() / Miii1();
                    // calculating the coefficients for formula (M are the octopole moments)
                    // the coefficients are calculated in (17) and (18)
                    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                        n0[j][i] = Miii1[j] - Miii0[j] * tmp1;
                        // n1[j][i] = Miii0[j] - Miii1[j] * tmp2;
                    }
                }
            }

            // distance between cells in all dimensions
            std::array<simd_vector, NDIM> dX;
#pragma GCC ivdep
            for (integer d = 0; d < NDIM; ++d) {
                dX[d] = X[d] - Y[d];
            }

            // R_i in paper is the dX in the code
            // D is taylor expansion value for a given X expansion of the gravitational
            // potential (multipole expansion)
            taylor<5, simd_vector> D;
            // // A0, A1 are the contributions to L (for each cell)
            // taylor<4, simd_vector> A1;
            // // B0, B1 are the contributions to L_c (for each cell)
            // std::array<simd_vector, NDIM> B1 = {
            //     simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO)};

            // calculates all D-values, calculate all coefficients of 1/r (not the potential),
            // formula (6)-(9) and (19)
            D.set_basis(dX);

            // the following loops calculate formula (10), potential from A->B and B->A
            // (basically alternating)
            A0[0] += m0[0] * D[0];
            // A1[0] += m1[0] * D[0];
            if (type != RHO) {
                for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
                    A0[0] -= m0[i] * D[i];
                    // A1[0] += m1[i] * D[i];
                }
            }
            for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
                const auto tmp = D[i] * (factor[i] * HALF);
                A0[0] += m0[i] * tmp;
                // A1[0] += m1[i] * tmp;
            }
            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                const auto tmp = D[i] * (factor[i] * SIXTH);
                A0[0] -= m0[i] * tmp;
                // A1[0] += m1[i] * tmp;
            }

            for (integer a = 0; a < NDIM; ++a) {
                int const* ab_idx_map = to_ab_idx_map3[a];
                int const* abc_idx_map = to_abc_idx_map3[a];

                A0(a) += m0() * D(a);
                // A1(a) += -m1() * D(a);
                for (integer i = 0; i != 6; ++i) {
                    if (type != RHO && i < 3) {
                        A0(a) -= m0(a) * D[ab_idx_map[i]];
                        // A1(a) -= m1(a) * D[ab_idx_map[i]];
                    }
                    const integer cb_idx = cb_idx_map[i];
                    const auto tmp1 = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                    A0(a) += m0[cb_idx] * tmp1;
                    // A1(a) -= m1[cb_idx] * tmp1;
                }
            }

            if (type == RHO) {
                for (integer a = 0; a != NDIM; ++a) {
                    int const* abcd_idx_map = to_abcd_idx_map3[a];
                    for (integer i = 0; i != 10; ++i) {
                        const integer bcd_idx = bcd_idx_map[i];
                        const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                        B0[a] -= n0[bcd_idx] * tmp;
                        // B1[a] -= n1[bcd_idx] * tmp;
                    }
                }
            }

            for (integer i = 0; i != 6; ++i) {
                int const* abc_idx_map6 = to_abc_idx_map6[i];

                integer const ab_idx = ab_idx_map6[i];
                A0[ab_idx] += m0() * D[ab_idx];
                // A1[ab_idx] += m1() * D[ab_idx];
                for (integer c = 0; c < NDIM; ++c) {
                    const auto& tmp = D[abc_idx_map6[c]];
                    A0[ab_idx] -= m0(c) * tmp;
                    // A1[ab_idx] += m1(c) * tmp;
                }
            }

            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                const auto& tmp = D[i];
                // A1[i] += -m1[0] * tmp;
                A0[i] += m0[0] * tmp;
            }

            /////////////////////////////////////////////////////////////////////////
            // potential and correction have been computed, now scatter the results
            /////////////////////////////////////////////////////////////////////////

            // for (integer i = 0; i != simd_len && i + interaction_second_index < inner_loop_stop;
            // ++i) {
            // for (integer i = 0; i < simd_len && i + interaction_second_index < inner_loop_stop;
            //      ++i) {    // TODO: for debugging
            //     const integer iii1 = interaction_list[interaction_second_index + i].second;

            //     for (integer j = 0; j != taylor_sizes[3]; ++j) {
            //         L[iii1][j] += A1[j][i];
            //     }

            //     if (type == RHO && opts.ang_con) {
            //         space_vector& L_ciii1 = L_c[iii1];
            //         for (integer j = 0; j != NDIM; ++j) {
            //             L_ciii1[j] += B1[j][i];
            //         }
            //     }
            // }
        }

        // for (integer i = 0; i != simd_len; ++i) {
        // const integer iii0 = interaction_list[interaction_first_index].first;
        multipole& Liii0 = L[iii0];
        // now add up the simd lanes

        // for (integer i = 0; i < simd_len; ++i) {
        for (integer j = 0; j != taylor_sizes[3]; ++j) {
#if Vc_IS_VERSION_2 == 0
            Liii0[j] += A0[j].sum();
#else
            Liii0[j] += Vc::reduce(A0[j]);
#endif
        }
        // }
        if (type == RHO && opts.ang_con) {
            space_vector& L_ciii0 = L_c[iii0];
            // for (integer i = 0; i < simd_len; ++i) {
            for (integer j = 0; j != NDIM; ++j) {
#if Vc_IS_VERSION_2 == 0
                L_ciii0[j] += B0[j].sum();
#else
                L_ciii0[j] += Vc::reduce(B0[j]);
#endif
            }
            // }
        }

        interaction_first_index = inner_loop_stop;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "compute_interactions_inner duration (ms): " << duration.count() << std::endl;
}
