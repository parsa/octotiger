#include "grid.hpp"
#include "grid_flattened_indices.hpp"
#include "options.hpp"
#include "profiler.hpp"
#include "simd.hpp"
#include "taylor.hpp"
#include "physcon.hpp"


#include <hpx/include/parallel_for_loop.hpp>

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

#ifdef USE_GRAV_PAR
const auto for_loop_policy = hpx::parallel::execution::par;
#else
const auto for_loop_policy = hpx::parallel::execution::seq;
#endif

void grid::compute_interactions(gsolve_type type) {
    PROF_BEGIN;

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
    if (!is_leaf) {
        // Non-leaf means that multipole-multipole interaction (David)
        // Fetch the interaction list for the current cell (David)
        const auto& this_ilist = is_root ? ilist_r : ilist_n;
        const integer list_size = this_ilist.size();

        // Space vector is a vector pack (David)
        // Center of masses of each cell (com_ptr1 is the center of mass of the parent cell)
        std::vector<space_vector> const& com0 = (*(com_ptr[0]));
        // Do 8 cell-cell interactions at the same time (simd.hpp) (Dominic)
        // This could lead to problems either on Haswell or on KNL as the number is hardcoded
        // (David)
        // It is unclear what the underlying vectorization library does, if simd_len > architectural
        // length (David)
        hpx::parallel::for_loop_strided(for_loop_policy, 0, list_size, simd_len,
            [&com0, &this_ilist, list_size, type, this](std::size_t li) {

                // dX is distance between X and Y
                // X and Y are the two cells interacting
                // X and Y store the 3D center of masses (per simd element, SoA style)
                std::array<simd_vector, NDIM> dX;
                std::array<simd_vector, NDIM> X;
                std::array<simd_vector, NDIM> Y;

                // m multipole moments of the cells
                taylor<4, simd_vector> m0;
                taylor<4, simd_vector> m1;
                // n angular momentum of the cells
                taylor<4, simd_vector> n0;
                taylor<4, simd_vector> n1;

                // vector of multipoles (== taylor<4>s)
                // here the multipoles are used as a storage of expansion coefficients
                auto& M = *M_ptr;

                // FIXME: replace with vector-pack gather-loads
                // TODO: only uses first 10 coefficients? ask Dominic
#ifndef __GNUG__
#pragma GCC ivdep
#endif
                for (integer i = 0; i != simd_len && integer(li + i) < list_size; ++i) {
                    const integer iii0 = this_ilist[li + i].first;
                    const integer iii1 = this_ilist[li + i].second;
                    space_vector const& com0iii0 = com0[iii0];
                    space_vector const& com0iii1 = com0[iii1];
                    for (integer d = 0; d < NDIM; ++d) {
                        // load the 3D center of masses
                        X[d][i] = com0iii0[d];
                        Y[d][i] = com0iii1[d];
                    }

                    // cell specific taylor series coefficients
                    multipole const& Miii0 = M[iii0];
                    multipole const& Miii1 = M[iii1];
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        m0[j][i] = Miii1[j];
                        m1[j][i] = Miii0[j];
                    }

                    // RHO computes the gravitational potential based on the mass density
                    // non-RHO computes the time derivative (hydrodynamics code V_t)
                    if (type == RHO) {
                        // this branch computes the angular momentum correction, (20) in the paper
                        // divide by mass of other cell
                        real const tmp1 = Miii1() / Miii0();
                        real const tmp2 = Miii0() / Miii1();
                        // calculating the coefficients for formula (M are the octopole moments)
                        // the coefficients are calculated in (17) and (18)
                        // TODO: Dominic
                        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                            n0[j][i] = Miii1[j] - Miii0[j] * tmp1;
                            n1[j][i] = Miii0[j] - Miii1[j] * tmp2;
                        }
                    }
                }

                if (type != RHO) {
#pragma GCC ivdep
                    for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                        n0[j] = ZERO;
                        n1[j] = ZERO;
                    }
                }

// distance between cells in all dimensions
#pragma GCC ivdep
                for (integer d = 0; d < NDIM; ++d) {
                    dX[d] = X[d] - Y[d];
                }

                // R_i in paper is the dX in the code
                // D is taylor expansion value for a given X expansion of the gravitational
                // potential (multipole expansion)
                taylor<5, simd_vector> D;
                // A0, A1 are the contributions to L (for each cell)
                taylor<4, simd_vector> A0, A1;
                // B0, B1 are the contributions to L_c (for each cell)
                std::array<simd_vector, NDIM> B0 = {
                    simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO)};
                std::array<simd_vector, NDIM> B1 = {
                    simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO)};

                // calculates all D-values, calculate all coefficients of 1/r (not the potential),
                // formula (6)-(9) and (19)
                D.set_basis(dX);

                // the following loops calculate formula (10), potential from A->B and B->A
                // (basically alternating)
                A0[0] = m0[0] * D[0];
                A1[0] = m1[0] * D[0];
                if (type != RHO) {
                    for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
                        A0[0] -= m0[i] * D[i];
                        A1[0] += m1[i] * D[i];
                    }
                }
                for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
                    const auto tmp = D[i] * (factor[i] * HALF);
                    A0[0] += m0[i] * tmp;
                    A1[0] += m1[i] * tmp;
                }
                for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                    const auto tmp = D[i] * (factor[i] * SIXTH);
                    A0[0] -= m0[i] * tmp;
                    A1[0] += m1[i] * tmp;
                }

//                 for (integer a = 0; a < NDIM; ++a) {
//                     A0(a) = m0() * D(a);
//                     A1(a) = -m1() * D(a);
//                     for (integer b = 0; b < NDIM; ++b) {
//                         if (type != RHO) {
//                             A0(a) -= m0(a) * D(a, b);
//                             A1(a) -= m1(a) * D(a, b);
//                         }
//                         for (integer c = b; c < NDIM; ++c) {
//                             const auto tmp1 = D(a, b, c) * (factor(c, b) * HALF);
//                             A0(a) += m0(c, b) * tmp1;
//                             A1(a) -= m1(c, b) * tmp1;
//                         }
//                     }
//                 }
                for (integer a = 0; a < NDIM; ++a) {
                    int const* ab_idx_map = to_ab_idx_map3[a];
                    int const* abc_idx_map = to_abc_idx_map3[a];

                    A0(a) = m0() * D(a);
                    A1(a) = -m1() * D(a);
                    for (integer i = 0; i != 6; ++i) {
                        if (type != RHO && i < 3) {
                            A0(a) -= m0(a) * D[ab_idx_map[i]];
                            A1(a) -= m1(a) * D[ab_idx_map[i]];
                        }
                        const integer cb_idx = cb_idx_map[i];
                        const auto tmp1 = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                        A0(a) += m0[cb_idx] * tmp1;
                        A1(a) -= m1[cb_idx] * tmp1;
                    }
                }

                if (type == RHO) {
//                     for (integer a = 0; a != NDIM; ++a) {
//                         for (integer b = 0; b != NDIM; ++b) {
//                             for (integer c = b; c != NDIM; ++c) {
//                                 for (integer d = c; d != NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                     B1[a] -= n1(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
                    for (integer a = 0; a != NDIM; ++a) {
                        int const* abcd_idx_map = to_abcd_idx_map3[a];
                        for (integer i = 0; i != 10; ++i) {
                            const integer bcd_idx = bcd_idx_map[i];
                            const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                            B0[a] -= n0[bcd_idx] * tmp;
                            B1[a] -= n1[bcd_idx] * tmp;
                        }
                    }
                }

//                 for (integer a = 0; a < NDIM; ++a) {
//                     for (integer b = a; b < NDIM; ++b) {
//                         A0(a, b) = m0() * D(a, b);
//                         A1(a, b) = m1() * D(a, b);
//                         for (integer c = 0; c < NDIM; ++c) {
//                             const auto& tmp = D(a, b, c);
//                             A0(a, b) -= m0(c) * tmp;
//                             A1(a, b) += m1(c) * tmp;
//                         }
//                     }
//                 }
                for (integer i = 0; i != 6; ++i) {
                    int const* abc_idx_map6 = to_abc_idx_map6[i];

                    integer const ab_idx = ab_idx_map6[i];
                    A0[ab_idx] = m0() * D[ab_idx];
                    A1[ab_idx] = m1() * D[ab_idx];
                    for (integer c = 0; c < NDIM; ++c) {
                        const auto& tmp = D[abc_idx_map6[c]];
                        A0[ab_idx] -= m0(c) * tmp;
                        A1[ab_idx] += m1(c) * tmp;
                    }
                }

                for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                    const auto& tmp = D[i];
                    A1[i] = -m1[0] * tmp;
                    A0[i] = m0[0] * tmp;
                }
#ifdef USE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif

                // potential and correction have been computed, no scatter the results
                for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
                    const integer iii0 = this_ilist[li + i].first;
                    const integer iii1 = this_ilist[li + i].second;

                    expansion tmp1, tmp2;
#pragma GCC ivdep
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        tmp1[j] = A0[j][i];
                        tmp2[j] = A1[j][i];
                    }
                    L[iii0] += tmp1;
                    L[iii1] += tmp2;

                    if (type == RHO && opts.ang_con) {
                        space_vector& L_ciii0 = L_c[iii0];
                        space_vector& L_ciii1 = L_c[iii1];
                        for (integer j = 0; j != NDIM; ++j) {
                            L_ciii0[j] += B0[j][i];
                            L_ciii1[j] += B1[j][i];
                        }
                    }
                }
            });
    } else {
#if !defined(HPX_HAVE_DATAPAR)
        const v4sd d0 = {1.0 / dx, +1.0 / sqr(dx), +1.0 / sqr(dx), +1.0 / sqr(dx)};
        const v4sd d1 = {1.0 / dx, -1.0 / sqr(dx), -1.0 / sqr(dx), -1.0 / sqr(dx)};
#else

        // first compute potential and the the three components of the force
        // vectorizing across the dimensions, therefore 1 component for the potential + 3
        // components for the force

        // Coefficients for potential evaluation? (David)
        const std::array<double, 4> di0 = {
            1.0 / dx, +1.0 / sqr(dx), +1.0 / sqr(dx), +1.0 / sqr(dx)};
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
        const v4sd d0(di0.data());
#else
        const v4sd d0(di0.data(), Vc::flags::vector_aligned);
#endif

        // negative of d0 because it's the force in the opposite direction
        const std::array<double, 4> di1 = {
            1.0 / dx, -1.0 / sqr(dx), -1.0 / sqr(dx), -1.0 / sqr(dx)};
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
        const v4sd d1(di1.data());
#else
        const v4sd d1(di1.data(), Vc::flags::vector_aligned);
#endif

#endif

        // Number of body-body interactions current leaf cell, probably includes interactions with
        // bodies in neighboring cells  (David)
        const integer dsize = ilist_d.size();

        // Iterate interaction description between bodies in the current cell and closeby bodies
        // (David)
        for (integer li = 0; li < dsize; ++li) {
            // Retrieve the interaction description for the current body (David)
            const auto& ele = ilist_d[li];
            // Extract the indices of the two interacting bodies (David)
            const integer iii0 = ele.first;
            const integer iii1 = ele.second;
#if !defined(HPX_HAVE_DATAPAR)
            v4sd m0, m1;
#pragma GCC ivdep
            for (integer i = 0; i != 4; ++i) {
                m0[i] = mon[iii1];
            }
#pragma GCC ivdep
            for (integer i = 0; i != 4; ++i) {
                m1[i] = mon[iii0];
            }

#ifdef USE_GRAV_PAR
            std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
            L[iii0] += m0 * ele.four * d0;
            L[iii1] += m1 * ele.four * d1;
#else
#ifdef USE_GRAV_PAR
            std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
            // fetch both interacting bodies (monopoles) (David)
            // broadcasts a single value
            L[iii0] += mon[iii1] * ele.four * d0;
            L[iii1] += mon[iii0] * ele.four * d1;
#endif
        }
    }
    PROF_END;
}
