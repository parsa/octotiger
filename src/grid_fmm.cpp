/*
 * grid_fmm.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: dmarce1
 */
#include "grid.hpp"
#include "options.hpp"
#include "profiler.hpp"
#include "simd.hpp"
#include "taylor.hpp"
#include "physcon.hpp"
#include "grid_fmm_new.hpp"

#include <hpx/include/parallel_for_loop.hpp>

#include <cstddef>
#include <utility>

constexpr integer TILE_WIDTH = 64;

#ifdef USE_GRAV_PAR
const auto for_loop_policy = hpx::parallel::execution::par;
#else
const auto for_loop_policy = hpx::parallel::execution::seq;
#endif

// why not lists per cell? (David)
// is a grid actually a cell? (David)
// what is the semantics of a space_vector? (David)

// non-root and non-leaf node (David)
static std::vector<interaction_type> ilist_n;
// monopole-monopole interactions? (David)
static std::vector<interaction_type> ilist_d;
// interactions of root node (empty list? boundary stuff?) (David)
static std::vector<interaction_type> ilist_r;
static std::vector<std::vector<boundary_interaction_type>> ilist_d_bnd(geo::direction::count());
static std::vector<std::vector<boundary_interaction_type>> ilist_n_bnd(geo::direction::count());
   
extern options opts;

//template <class T>
void load_multipole(taylor<4, simd_vector>& m, space_vector& c, const gravity_boundary_type& data,
    integer iter, bool monopole) {
    if (monopole) {
        m = simd_vector(0.0);
        #warning This is probably broken, FIXME
        m = simd_vector((*(data.m))[iter]);
    } else {
        //auto const& tmp1 = (*(data.M))[iter];
#pragma GCC ivdep
        for (int j = 0; j != 20; ++j) {
            for (int i = 0; i != 8; ++i)
                m[j][i] = (*(data.M))[j][iter]; // FIXME: Inefficient access.
        }
        //auto const& tmp2 = (*(data.x))[iter];
#pragma GCC ivdep
        for (integer d = 0; d != NDIM; ++d) {
            c[d] = (*(data.com))[d][iter]; // FIXME: Inefficient access.
        }
    }
}

void find_eigenvectors(real q[3][3], real e[3][3], real lambda[3]) {
    PROF_BEGIN;
    real b0[3], b1[3], A, bdif;
    int iter = 0;
    for (int l = 0; l < 3; l++) {
        b0[0] = b0[1] = b0[2] = 0.0;
        b0[l] = 1.0;
        do {
            iter++;
            b1[0] = b1[1] = b1[2] = 0.0;
            for (int i = 0; i < 3; i++) {
                for (int m = 0; m < 3; m++) {
                    b1[i] += q[i][m] * b0[m];
                }
            }
            A = sqrt(sqr(b1[0]) + sqr(b1[1]) + sqr(b1[2]));
            bdif = 0.0;
            for (int i = 0; i < 3; i++) {
                b1[i] = b1[i] / A;
                bdif += pow(b0[i] - b1[i], 2);
            }
            for (int i = 0; i < 3; i++) {
                b0[i] = b1[i];
            }

        } while (fabs(bdif) > 1.0e-14);
        for (int m = 0; m < 3; m++) {
            e[l][m] = b0[m];
        }
        for (int i = 0; i < 3; i++) {
            A += b0[i] * q[l][i];
        }
        lambda[l] = sqrt(A) / sqrt(sqr(e[l][0]) + sqr(e[l][1]) + sqr(e[l][2]));
    }
    PROF_END;
}

std::pair<space_vector, space_vector> grid::find_axis() const {
    PROF_BEGIN;
    real quad_moment[NDIM][NDIM];
    real eigen[NDIM][NDIM];
    real lambda[NDIM];
    space_vector this_com;
    real mtot = 0.0;
    for (integer i = 0; i != NDIM; ++i) {
        this_com[i] = 0.0;
        for (integer j = 0; j != NDIM; ++j) {
            quad_moment[i][j] = 0.0;
        }
    }

    auto& M = *M_ptr;
    auto& mon = *mon_ptr;
    auto const& com0 = *com0_ptr;
    for (integer i = 0; i != G_NX; ++i) {
        for (integer j = 0; j != G_NX; ++j) {
            for (integer k = 0; k != G_NX; ++k) {
                const integer iii1 = gindex(i, j, k);
                const integer iii0 = gindex(i + H_BW, j + H_BW, k + H_BW);
                for (integer n = 0; n != NDIM; ++n) {
                    real mass;
                    if (is_leaf) {
                        mass = mon[iii1];
                    } else {
                        mass = M()[iii1];
                    }
                    this_com[n] += mass * com0[n][iii1];
                    mtot += mass;
                    for (integer m = 0; m != NDIM; ++m) {
                        if (!is_leaf) {
                            quad_moment[n][m] += M(n, m)[iii1];
                        }
                        quad_moment[n][m] += mass * com0[n][iii1] * com0[m][iii1];
                    }
                }
            }
        }
    }
    for (integer j = 0; j != NDIM; ++j) {
        this_com[j] /= mtot;
    }

    find_eigenvectors(quad_moment, eigen, lambda);
    integer index;
    if (lambda[0] > lambda[1] && lambda[0] > lambda[2]) {
        index = 0;
    } else if (lambda[1] > lambda[2]) {
        index = 1;
    } else {
        index = 2;
    }
    space_vector rc;
    for (integer j = 0; j != NDIM; ++j) {
        rc[j] = eigen[index][j];
    }
    std::pair<space_vector, space_vector> pair{rc, this_com};
    PROF_END;
    return pair;
}

compute_interactions_stats_t grid::compute_interactions(gsolve_type type) {
    if (!is_leaf && !opts.compute_interactions_legacy)
    {
        if (is_root)
        {
            if (RHO == type)
            {
                if (opts.ang_con)
                    // Non-leaf, root, ANG_CON_ON, RHO
                    return compute_interactions_non_leaf<&ilist_r, TILE_WIDTH, ANG_CON_ON, RHO>();
                else
                    // Non-leaf, root, ANG_CON_OFF, RHO
                    return compute_interactions_non_leaf<&ilist_r, TILE_WIDTH, ANG_CON_OFF, NON_RHO>();
            }
            else
            {
                if (opts.ang_con)
                    // Non-leaf, root, non-ANG_CON_ON, RHO
                    return compute_interactions_non_leaf<&ilist_r, TILE_WIDTH, ANG_CON_ON, NON_RHO>();
                else
                    // Non-leaf, root, non-ANG_CON_OFF, RHO
                    return compute_interactions_non_leaf<&ilist_r, TILE_WIDTH, ANG_CON_OFF, NON_RHO>();
            }
        }
        else
        {
            if (RHO == type)
            {
                if (opts.ang_con)
                    // Non-leaf, non-root, ANG_CON_ON, RHO
                    return compute_interactions_non_leaf<&ilist_n, TILE_WIDTH, ANG_CON_ON, RHO>();
                else
                    // Non-leaf, non-root, ANG_CON_OFF, RHO
                    return compute_interactions_non_leaf<&ilist_n, TILE_WIDTH, ANG_CON_OFF, RHO>();
            }
            else
            {
                if (opts.ang_con)
                    // Non-leaf, non-root, non-ANG_CON_ON, RHO
                    return compute_interactions_non_leaf<&ilist_n, TILE_WIDTH, ANG_CON_ON, NON_RHO>();
                else
                    // Non-leaf, non-root, non-ANG_CON_OFF, RHO
                    return compute_interactions_non_leaf<&ilist_n, TILE_WIDTH, ANG_CON_OFF, NON_RHO>();
            }
        }
    }
    else
    {
        // Leaf
        compute_interactions_legacy(type);
        return compute_interactions_stats_t{};
    }
}

void grid::solve_gravity(gsolve_type type) {
    compute_multipoles(type);
    compute_interactions(type);
    compute_expansions(type);
}

constexpr int to_ab_idx_map3[3][3] = {
    {  4,  5,  6  },
    {  5,  7,  8  },
    {  6,  8,  9  }
};

constexpr int cb_idx_map[6] = {
     4,  5,  6,  7,  8,  9,
};

constexpr int to_abc_idx_map3[3][6] = {
    {  10, 11, 12, 13, 14, 15, },
    {  11, 13, 14, 16, 17, 18, },
    {  12, 14, 15, 17, 18, 19, }
};

constexpr int to_abcd_idx_map3[3][10] = {
    { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 },
    { 21, 23, 24, 26, 27, 28, 30, 31, 32, 33 },
    { 22, 24, 25, 27, 28, 29, 31, 32, 33, 34 }
};

constexpr int bcd_idx_map[10] = {
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19
};

constexpr int to_abc_idx_map6[6][3] = {
    { 10, 11, 12 }, { 11, 13, 14 },
    { 12, 14, 15 }, { 13, 16, 17 },
    { 14, 17, 18 }, { 15, 18, 19 }
};

constexpr int ab_idx_map6[6] = {
    4, 5, 6, 7, 8, 9
};

void grid::compute_interactions_legacy(gsolve_type type) {
    PROF_BEGIN;

    // calculating the contribution of all the inner cells
    // calculating the interaction

    auto& mon = *mon_ptr;
    // L stores the gravitational potential
    // L should be L, (10) in the paper
    // L_c stores the correction for angular momentum
    // L_c, (20) in the paper (Dominic)
    for (auto& a : L_PLACEHOLDER)
        std::fill(a.begin(), a.end(), ZERO);

    if (opts.ang_con) {
        for (auto& a : L_c_PLACEHOLDER)
            std::fill(a.begin(), a.end(), ZERO);
    }

    // Non-leaf nodes use taylor expansion
    // For leaf node, calculates the gravitational potential between two particles
    if (!is_leaf) {
        // Non-leaf means that multipole-multipole interaction (David)
        // Fetch the interaction list for the current cell (David)
        const auto& this_ilist = is_root ? ilist_r : ilist_n;
        const integer list_size = this_ilist.size();

        // Space vector is a vector pack (David)
        // Center of masses of each cell (com1_ptr is the center of mass of the parent cell)
        auto const& com0 = *com0_ptr;

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

                    // load the 3D center of masses
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][i] = com0[d][iii0];
                        Y[d][i] = com0[d][iii1];
                    }

                    // load cell specific taylor series coefficients
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        m0[j][i] = M[j][iii1];
                        m1[j][i] = M[j][iii0];
                    }

                    // RHO computes the gravitational potential based on the mass density
                    // non-RHO computes the time derivative (hydrodynamics code V_t)
                    if (type == RHO) {
                        // this branch computes the angular momentum correction, (20) in the paper
                        // divide by mass of other cell
                        real const tmp1 = M()[iii1] / M()[iii0];
                        real const tmp2 = M()[iii0] / M()[iii1];
                        // calculating the coefficients for formula (M are the octopole moments)
                        // the coefficients are calculated in (17) and (18)
                        // TODO: Dominic
                        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                            n0[j][i] = M[j][iii1] - M[j][iii0] * tmp1;
                            n1[j][i] = M[j][iii0] - M[j][iii1] * tmp2;
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

#pragma GCC ivdep
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        L_PLACEHOLDER[j][iii0] += A0[j][i];
                        L_PLACEHOLDER[j][iii1] += A1[j][i];
                    }

                    if (type == RHO && opts.ang_con) {
                        for (integer j = 0; j != NDIM; ++j) {
                            L_c_PLACEHOLDER[j][iii0] += B0[j][i];
                            L_c_PLACEHOLDER[j][iii1] += B1[j][i];
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
            //L[iii0] += m0 * ele.four * d0;
            //L[iii1] += m1 * ele.four * d1;
            for (integer x = 0; x != 4; ++x)
            {
                L_PLACEHOLDER[x][iii0] += m0[x] * ele.four[x] * d0[x];
                L_PLACEHOLDER[x][iii1] += m1[x] * ele.four[x] * d1[x];
            }
#else
#ifdef USE_GRAV_PAR
            std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
            // fetch both interacting bodies (monopoles) (David)
            // broadcasts a single value

            // Originally: L[iii0]           +=  mon[iii1] * ele.four          * d0
            //             L[iii1]           +=  mon[iii0] * ele.four          * d1
            //
            //             expansion         +=  double    * v4sd              * v4sd
            //             expansion         += (double    * v4sd            ) * v4sd                [Precedence]
            //             taylor<4, double> += (double    * v4sd            ) * v4sd
            //             array<double, 20> += (double    * array<double, 4>) * array<double, 4>
            //             array<double, 20> += (array<double, 4>            ) * array<double, 4>    [Evaluate #1]
            //             array<double, 20> += array<double, 4>                                     [Evaluate #2]
            for (integer x = 0; x != 4; ++x)
            {
                L_PLACEHOLDER[x][iii0] += mon[iii1] * ele.four[x] * d0[x];
                L_PLACEHOLDER[x][iii1] += mon[iii0] * ele.four[x] * d1[x];
            }
#endif
        }
    }
    PROF_END;
}

void grid::compute_boundary_interactions(gsolve_type type, const geo::direction& dir,
    bool is_monopole, const gravity_boundary_type& mpoles) {
    if (!is_leaf) {
        if (!is_monopole) {
            compute_boundary_interactions_multipole_multipole(type, ilist_n_bnd[dir], mpoles);
        } else {
            compute_boundary_interactions_monopole_multipole(type, ilist_d_bnd[dir], mpoles);
        }
    } else {
        if (!is_monopole) {
            compute_boundary_interactions_multipole_monopole(type, ilist_d_bnd[dir], mpoles);
        } else {
            compute_boundary_interactions_monopole_monopole(type, ilist_d_bnd[dir], mpoles);
        }
    }
}

void grid::compute_boundary_interactions_multipole_multipole(gsolve_type type,
    const std::vector<boundary_interaction_type>& ilist_n_bnd,
    const gravity_boundary_type& mpoles) {
    PROF_BEGIN;
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

    auto const& com0 = *com0_ptr;
    hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(),
        [&mpoles, &com0, &ilist_n_bnd, type, this, &M](std::size_t si) {

            taylor<4, simd_vector> m0;
            taylor<4, simd_vector> n0;
            std::array<simd_vector, NDIM> dX;
            std::array<simd_vector, NDIM> X;
            space_vector Y;

            boundary_interaction_type const& bnd = ilist_n_bnd[si];
            integer index = mpoles.is_local ? bnd.second : si;

            load_multipole(m0, Y, mpoles, index, false);

            std::array<simd_vector, NDIM> simdY = {
                simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])),
            };

            const integer list_size = bnd.first.size();
            for (integer li = 0; li < list_size; li += simd_len) {
#ifndef __GNUG__
#pragma GCC ivdep
#endif
            	for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
                    const integer iii0 = bnd.first[li + i];
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][i] = com0[d][iii0];
                    }
                    if (type == RHO) {
                        real const tmp = m0()[i] / M()[iii0];
                        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                            n0[j][i] = m0[j][i] - M[j][iii0] * tmp;
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
                    dX[d] = X[d] - simdY[d];
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

//                 for (integer a = 0; a < NDIM; ++a) {
//                     A0(a) = m0() * D(a);
//                     for (integer b = 0; b < NDIM; ++b) {
//                         if (type != RHO) {
//                             A0(a) -= m0(a) * D(a, b);
//                         }
//                         for (integer c = b; c < NDIM; ++c) {
//                             const auto tmp = D(a, b, c) * (factor(c, b) * HALF);
//                             A0(a) += m0(c, b) * tmp;
//                         }
//                     }
//                 }
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
//                     for (integer a = 0; a < NDIM; ++a) {
//                         for (integer b = 0; b < NDIM; ++b) {
//                             for (integer c = b; c < NDIM; ++c) {
//                                 for (integer d = c; d < NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
                    for (integer a = 0; a < NDIM; ++a) {
                        int const* abcd_idx_map = to_abcd_idx_map3[a];
                        for (integer i = 0; i != 10; ++i) {
                            const integer bcd_idx = bcd_idx_map[i];
                            const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                            B0[a] -= n0[bcd_idx] * tmp;
                        }
                    }
                }

//                 for (integer a = 0; a < NDIM; ++a) {
//                     for (integer b = a; b < NDIM; ++b) {
//                         A0(a, b) = m0() * D(a, b);
//                         for (integer c = 0; c < NDIM; ++c) {
//                             A0(a, b) -= m0(c) * D(a, b, c);
//                         }
//                     }
//                 }
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

#ifdef USE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
                for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
                    const integer iii0 = bnd.first[li + i];

#pragma GCC ivdep
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        L_PLACEHOLDER[j][iii0] += A0[j][i];
                    }

                    if (type == RHO && opts.ang_con) {
#pragma GCC ivdep
                        for (integer j = 0; j != NDIM; ++j) {
                            L_c_PLACEHOLDER[j][iii0] += B0[j][i];
                        }
                    }
                }
            }
        });
    PROF_END;
}

void grid::compute_boundary_interactions_multipole_monopole(gsolve_type type,
    const std::vector<boundary_interaction_type>& ilist_n_bnd,
    const gravity_boundary_type& mpoles) {
    PROF_BEGIN;
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

    auto const& com0 = *com0_ptr;
    hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(),
        [&mpoles, &com0, &ilist_n_bnd, type, this](std::size_t si) {

            taylor<4, simd_vector> m0;
            taylor<4, simd_vector> n0;
            std::array<simd_vector, NDIM> dX;
            std::array<simd_vector, NDIM> X;
            space_vector Y;

            boundary_interaction_type const& bnd = ilist_n_bnd[si];
            const integer list_size = bnd.first.size();
            integer index = mpoles.is_local ? bnd.second : si;
            load_multipole(m0, Y, mpoles, index, false);

            std::array<simd_vector, NDIM> simdY = {
                simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])),
            };

            if (type == RHO) {
#pragma GCC ivdep
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = m0[j];
                }
            } else {
#pragma GCC ivdep
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = ZERO;
                }
            }
#pragma GCC ivdep
            for (integer li = 0; li < list_size; li += simd_len) {
                for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
                    const integer iii0 = bnd.first[li + i];
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][i] = com0[d][iii0];
                    }
                }
#pragma GCC ivdep
                for (integer d = 0; d < NDIM; ++d) {
                    dX[d] = X[d] - simdY[d];
                }

                taylor<5, simd_vector> D;
                taylor<2, simd_vector> A0;
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

//                 for (integer a = 0; a < NDIM; ++a) {
//                     A0(a) = m0() * D(a);
//                     for (integer b = 0; b < NDIM; ++b) {
//                         if (type != RHO) {
//                             A0(a) -= m0(a) * D(a, b);
//                         }
//                         for (integer c = b; c < NDIM; ++c) {
//                             A0(a) += m0(c, b) * D(a, b, c) * (factor(b, c) * HALF);
//                         }
//                     }
//                 }
                for (integer a = 0; a < NDIM; ++a) {
                    int const* ab_idx_map = to_ab_idx_map3[a];
                    int const* abc_idx_map = to_abc_idx_map3[a];

                    A0(a) = m0() * D(a);
                    for (integer i = 0; i != 6; ++i) {
                        if (type != RHO && i < 3) {
                            A0(a) -= m0(a) * D[ab_idx_map[i]];
                        }
                        const integer cb_idx = cb_idx_map[i];
                        A0(a) += m0[cb_idx] * D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                    }
                }

                if (type == RHO) {
//                     for (integer a = 0; a < NDIM; ++a) {
//                         for (integer b = 0; b < NDIM; ++b) {
//                             for (integer c = b; c < NDIM; ++c) {
//                                 for (integer d = c; d < NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
//                                 }
//                             }
//                         }
//                     }
                    for (integer a = 0; a < NDIM; ++a) {
                        int const* abcd_idx_map = to_abcd_idx_map3[a];
                        for (integer i = 0; i != 10; ++i) {
                            const integer bcd_idx = bcd_idx_map[i];
                            const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                            B0[a] -= n0[bcd_idx] * tmp;
                        }
                    }
                }

#ifdef USE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
                for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
                    const integer iii0 = bnd.first[li + i];
#pragma GCC ivdep
                    for (integer j = 0; j != 4; ++j) {
                        L_PLACEHOLDER[j][iii0] += A0[j][i];
                    }
                    if (type == RHO && opts.ang_con) {
#pragma GCC ivdep
                        for (integer j = 0; j != NDIM; ++j) {
                            L_c_PLACEHOLDER[j][iii0] += B0[j][i];
                        }
                    }
                }
            }
        });
    PROF_END;
}

void grid::compute_boundary_interactions_monopole_multipole(gsolve_type type,
    const std::vector<boundary_interaction_type>& ilist_n_bnd,
    const gravity_boundary_type& mpoles) {
    PROF_BEGIN;
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

    std::array<real, NDIM> Xbase = {X[0][hindex(H_BW, H_BW, H_BW)], X[1][hindex(H_BW, H_BW, H_BW)],
        X[2][hindex(H_BW, H_BW, H_BW)]};

    auto const& com0 = *com0_ptr;
    hpx::parallel::for_loop(for_loop_policy, 0, ilist_n_bnd.size(),
        [&mpoles, &Xbase, &com0, &ilist_n_bnd, type, this, &M](std::size_t si) {

            simd_vector m0;
            taylor<4, simd_vector> n0;
            std::array<simd_vector, NDIM> dX;
            std::array<simd_vector, NDIM> X;
            std::array<simd_vector, NDIM> Y;

            boundary_interaction_type const& bnd = ilist_n_bnd[si];
            integer index = mpoles.is_local ? bnd.second : si;
            const integer list_size = bnd.first.size();

#pragma GCC ivdep
            for (integer d = 0; d != NDIM; ++d) {
                Y[d] = bnd.x[d] * dx + Xbase[d];
            }

            m0 = (*(mpoles.m))[index];
            for (integer li = 0; li < list_size; li += simd_len) {
                for (integer i = 0; i != simd_len && li + i < list_size; ++i) {
                    const integer iii0 = bnd.first[li + i];
#pragma GCC ivdep
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][i] = com0[d][iii0];
                    }
                    if (type == RHO) {
                        real const tmp = m0[i] / M()[iii0];
#pragma GCC ivdep
                        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                            n0[j][i] = -M[j][iii0] * tmp;
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
                    dX[d] = X[d] - Y[d];
                }

                taylor<5, simd_vector> D;
                taylor<4, simd_vector> A0;
                std::array<simd_vector, NDIM> B0 = {
                    simd_vector(0.0), simd_vector(0.0), simd_vector(0.0)};

                D.set_basis(dX);

                A0[0] = m0 * D[0];
                for (integer i = taylor_sizes[0]; i != taylor_sizes[2]; ++i) {
                    A0[i] = m0 * D[i];
                }
                if (type == RHO) {
                    for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                        A0[i] = m0 * D[i];
                    }
                }

                if (type == RHO) {
//                     for (integer a = 0; a != NDIM; ++a) {
//                         for (integer b = 0; b != NDIM; ++b) {
//                             for (integer c = b; c != NDIM; ++c) {
//                                 for (integer d = c; d != NDIM; ++d) {
//                                     const auto tmp = D(a, b, c, d) * (factor(b, c, d) * SIXTH);
//                                     B0[a] -= n0(b, c, d) * tmp;
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
                        }
                    }
                }

#ifdef USE_GRAV_PAR
                std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
                for (integer i = 0; i != simd_len && i + li < list_size; ++i) {
                    const integer iii0 = bnd.first[li + i];

#pragma GCC ivdep
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        L_PLACEHOLDER[j][iii0] += A0[j][i];
                    }

                    if (type == RHO && opts.ang_con) {
#pragma GCC ivdep
                        for (integer j = 0; j != NDIM; ++j) {
                            L_c_PLACEHOLDER[j][iii0] += B0[j][i];
                        }
                    }
                }
            }
        });
    PROF_END;
}

void grid::compute_boundary_interactions_monopole_monopole(gsolve_type type,
    const std::vector<boundary_interaction_type>& ilist_n_bnd,
    const gravity_boundary_type& mpoles) {
    PROF_BEGIN;
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

#if !defined(HPX_HAVE_DATAPAR)
    const v4sd d0 = {1.0 / dx, +1.0 / sqr(dx), +1.0 / sqr(dx), +1.0 / sqr(dx)};
#else
    const std::array<double, 4> di0 = {1.0 / dx, +1.0 / sqr(dx), +1.0 / sqr(dx), +1.0 / sqr(dx)};
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
    const v4sd d0(di0.data());
#else
    const v4sd d0(di0.data(), Vc::flags::vector_aligned);
#endif
#endif
    hpx::parallel::for_loop(
        for_loop_policy, 0, ilist_n_bnd.size(),
        [&mpoles, &ilist_n_bnd, &d0, this](std::size_t si) {

            boundary_interaction_type const& bnd = ilist_n_bnd[si];
            const integer dsize = bnd.first.size();
            integer index = mpoles.is_local ? bnd.second : si;
#if !defined(HPX_HAVE_DATAPAR)
            const auto tmp = (*(mpoles).m)[index];
            v4sd m0;
#pragma GCC ivdep
            for (integer i = 0; i != 4; ++i) {
                m0[i] = tmp;
            }
#else
                v4sd m0 = (*(mpoles).m)[index];
#endif
            m0 *= d0;
#ifdef USE_GRAV_PAR
            std::lock_guard<hpx::lcos::local::spinlock> lock(*L_mtx);
#endif
            for (integer li = 0; li < dsize; ++li) {
                const integer iii0 = bnd.first[li];
                auto tmp1 = m0 * bnd.four[li];
#pragma GCC ivdep
                for (integer x = 0; x != 4; ++x) {
                    L_PLACEHOLDER[x][iii0] += tmp1[x];
                }
            }
        });
    PROF_END;
}

void compute_ilist() {
    const integer inx = INX;
    const integer nx = inx;
    interaction_type np;
    interaction_type dp;
    std::vector<interaction_type> ilist_r0;
    std::vector<interaction_type> ilist_n0;
    std::vector<interaction_type> ilist_d0;
    std::array<std::vector<interaction_type>, geo::direction::count()> ilist_n0_bnd;
    std::array<std::vector<interaction_type>, geo::direction::count()> ilist_d0_bnd;
    const real theta0 = opts.theta;
    const auto theta = [](integer i0, integer j0, integer k0, integer i1, integer j1, integer k1) {
        real tmp = (sqr(i0 - i1) + sqr(j0 - j1) + sqr(k0 - k1));
        // protect against sqrt(0)
        if (tmp > 0.0) {
            return 1.0 / (std::sqrt(tmp));
        } else {
            return 1.0e+10;
        }
    };
    const auto interior = [](integer i, integer j, integer k) {
        bool rc = true;
        if (i < 0 || i >= G_NX) {
            rc = false;
        } else if (j < 0 || j >= G_NX) {
            rc = false;
        } else if (k < 0 || k >= G_NX) {
            rc = false;
        }
        return rc;
    };
    const auto neighbor_dir = [](integer i, integer j, integer k) {
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
    };
    integer max_d = 0;
    integer width = INX;
    for (integer i0 = 0; i0 != nx; ++i0) {
        for (integer j0 = 0; j0 != nx; ++j0) {
            for (integer k0 = 0; k0 != nx; ++k0) {
                integer this_d = 0;
                integer ilb = std::min(i0 - width, integer(0));
                integer jlb = std::min(j0 - width, integer(0));
                integer klb = std::min(k0 - width, integer(0));
                integer iub = std::max(i0 + width + 1, G_NX);
                integer jub = std::max(j0 + width + 1, G_NX);
                integer kub = std::max(k0 + width + 1, G_NX);
                for (integer i1 = ilb; i1 < iub; ++i1) {
                    for (integer j1 = jlb; j1 < jub; ++j1) {
                        for (integer k1 = klb; k1 < kub; ++k1) {
                            const real x = i0 - i1;
                            const real y = j0 - j1;
                            const real z = k0 - k1;
                            // protect against sqrt(0)
                            const real tmp = sqr(x) + sqr(y) + sqr(z);
                            const real r = (tmp == 0) ? 0 : std::sqrt(tmp);
                            const real r3 = r * r * r;
                            v4sd four;
                            if (r > 0.0) {
                                four[0] = -1.0 / r;
                                four[1] = x / r3;
                                four[2] = y / r3;
                                four[3] = z / r3;
                            } else {
                                for (integer i = 0; i != 4; ++i) {
                                    four[i] = 0.0;
                                }
                            }
                            if (i0 == i1 && j0 == j1 && k0 == k1) {
                                continue;
                            }
                            const integer i0_c = (i0 + INX) / 2 - INX / 2;
                            const integer j0_c = (j0 + INX) / 2 - INX / 2;
                            const integer k0_c = (k0 + INX) / 2 - INX / 2;
                            const integer i1_c = (i1 + INX) / 2 - INX / 2;
                            const integer j1_c = (j1 + INX) / 2 - INX / 2;
                            const integer k1_c = (k1 + INX) / 2 - INX / 2;
                            const real theta_f = theta(i0, j0, k0, i1, j1, k1);
                            const real theta_c = theta(i0_c, j0_c, k0_c, i1_c, j1_c, k1_c);
                            const integer iii0 = gindex(i0, j0, k0);
                            const integer iii1n =
                                gindex((i1 + INX) % INX, (j1 + INX) % INX, (k1 + INX) % INX);
                            const integer iii1 = gindex(i1, j1, k1);
                            if (theta_c > theta0 && theta_f <= theta0) {
                                np.first = iii0;
                                np.second = iii1n;
                                np.four = four;
                                np.x[XDIM] = i1;
                                np.x[YDIM] = j1;
                                np.x[ZDIM] = k1;
                                if (interior(i1, j1, k1) && interior(i0, j0, k0)) {
                   //                 if (iii1 > iii0) {
                                        ilist_n0.push_back(np);
                    //                }
                                } else if (interior(i0, j0, k0)) {
                                    ilist_n0_bnd[neighbor_dir(i1, j1, k1)].push_back(np);
                                }
                            }
                            if (theta_c > theta0) {
                                ++this_d;
                                dp.first = iii0;
                                dp.second = iii1n;
                                dp.x[XDIM] = i1;
                                dp.x[YDIM] = j1;
                                dp.x[ZDIM] = k1;
                                dp.four = four;
                                if (interior(i1, j1, k1) && interior(i0, j0, k0)) {
                                    if (iii1 > iii0) {
                                        ilist_d0.push_back(dp);
                                    }
                                } else if (interior(i0, j0, k0)) {
                                    ilist_d0_bnd[neighbor_dir(i1, j1, k1)].push_back(dp);
                                }
                            }
                            if (theta_f <= theta0) {
                                np.first = iii0;
                                np.second = iii1n;
                                np.x[XDIM] = i1;
                                np.x[YDIM] = j1;
                                np.x[ZDIM] = k1;
                                np.four = four;
                                if (interior(i1, j1, k1) && interior(i0, j0, k0)) {
                      //              if (iii1 > iii0) {
                                        ilist_r0.push_back(np);
                       //             }
                                }
                            }
                        }
                    }
                }
                max_d = std::max(max_d, this_d);
            }
        }
    }

//     printf("# direct = %i\n", int(max_d));
    ilist_n = std::vector<interaction_type>(ilist_n0.begin(), ilist_n0.end());
    ilist_d = std::vector<interaction_type>(ilist_d0.begin(), ilist_d0.end());
    ilist_r = std::vector<interaction_type>(ilist_r0.begin(), ilist_r0.end());
    for (auto& dir : geo::direction::full_set()) {
        auto& d = ilist_d_bnd[dir];
        auto& d0 = ilist_d0_bnd[dir];
        auto& n = ilist_n_bnd[dir];
        auto& n0 = ilist_n0_bnd[dir];
        for (auto i0 : d0) {
            bool found = false;
            for (auto& i : d) {
                if (i.second == i0.second) {
                    i.first.push_back(i0.first);
                    i.four.push_back(i0.four);
                    found = true;
                    break;
                }
            }
            if (!found) {
                boundary_interaction_type i;
                i.second = i0.second;
                i.x = i0.x;
                n.push_back(i);
                i.first.push_back(i0.first);
                i.four.push_back(i0.four);
                d.push_back(i);
            }
        }
        for (auto i0 : n0) {
            bool found = false;
            for (auto& i : n) {
                if (i.second == i0.second) {
                    i.first.push_back(i0.first);
                    i.four.push_back(i0.four);
                    found = true;
                    break;
                }
            }
            assert(found);
        }
    }
/*
    {
        std::int32_t cur_index = ilist_r[0].first;
        std::int32_t cur_index_start = 0;
        for (std::int32_t li = 0; li != ilist_r.size(); ++li) {
            if (ilist_r[li].first != cur_index) {
                ilist_r[cur_index_start].inner_loop_end = li;
                cur_index = ilist_r[li].first;
                cur_index_start = li;
            }
        }
        // make sure the last element is handled correctly as well
        ilist_r[cur_index_start].inner_loop_end = ilist_r.size();
    }

    {
        std::int32_t cur_index = ilist_n[0].first;
        std::int32_t cur_index_start = 0;
        for (std::int32_t li = 0; li != ilist_n.size(); ++li) {
            if (ilist_n[li].first != cur_index) {
                ilist_n[cur_index_start].inner_loop_end = li;
                cur_index = ilist_n[li].first;
                cur_index_start = li;
            }
        }
        // make sure the last element is handled correctly as well
        ilist_n[cur_index_start].inner_loop_end = ilist_n.size();
    }

    {
        std::int32_t cur_index = ilist_d[0].first;
        std::int32_t cur_index_start = 0;
        for (std::int32_t li = 0; li != ilist_d.size(); ++li) {
            if (ilist_d[li].first != cur_index) {
                ilist_d[cur_index_start].inner_loop_end = li;
                cur_index = ilist_d[li].first;
                cur_index_start = li;
            }
        }
        // make sure the last element is handled correctly as well
        ilist_d[cur_index_start].inner_loop_end = ilist_d.size();
    }
*/
}

expansion_pass_type grid::compute_expansions(
    gsolve_type type, const expansion_pass_type* parent_expansions) {
    PROF_BEGIN;
    expansion_pass_type exp_ret;
    if (!is_leaf) {
        exp_ret.first.resize(INX * INX * INX);
        if (type == RHO) {
            exp_ret.second.resize(INX * INX * INX);
        }
    }
    const integer inx = INX;
    const integer nxp = (inx / 2);
    auto child_index = [=](
        integer ip, integer jp, integer kp, integer ci, integer bw = 0) -> integer {
        const integer ic = (2 * (ip) + bw) + ((ci >> 0) & 1);
        const integer jc = (2 * (jp) + bw) + ((ci >> 1) & 1);
        const integer kc = (2 * (kp) + bw) + ((ci >> 2) & 1);
        return (inx + 2 * bw) * (inx + 2 * bw) * ic + (inx + 2 * bw) * jc + kc;
    };

    auto const& com0 = *com0_ptr;
    auto const& com1 = *com1_ptr;
    for (integer ip = 0; ip != nxp; ++ip) {
        for (integer jp = 0; jp != nxp; ++jp) {
            for (integer kp = 0; kp != nxp; ++kp) {
                const integer iiip = nxp * nxp * ip + nxp * jp + kp;
                std::array<simd_vector, NDIM> X;
                std::array<simd_vector, NDIM> dX;
                taylor<4, simd_vector> l;
                std::array<simd_vector, NDIM> lc;
                if (!is_root) {
                    const integer index = (INX * INX / 4) * (ip) + (INX / 2) * (jp) + (kp);
                    for (integer j = 0; j != 20; ++j) {
                        l[j] = parent_expansions->first[index][j];
                    }
                    if (type == RHO && opts.ang_con) {
                        for (integer j = 0; j != NDIM; ++j) {
                            lc[j] = parent_expansions->second[index][j];
                        }
                    }
                } else {
                    for (integer j = 0; j != 20; ++j) {
                        l[j] = 0.0;
                    }
                    if (opts.ang_con) {
                        for (integer j = 0; j != NDIM; ++j) {
                            lc[j] = 0.0;
                        }
                    }
                }
                for (integer ci = 0; ci != NCHILD; ++ci) {
                    const integer iiic = child_index(ip, jp, kp, ci);
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][ci] = com0[d][iiic];
                    }
                }
                for (integer d = 0; d < NDIM; ++d) {
                    dX[d] = X[d] - com1[d][iiip];
                }
                l <<= dX;
                for (integer ci = 0; ci != NCHILD; ++ci) {
                    const integer iiic = child_index(ip, jp, kp, ci);
                    for (integer j = 0; j != 20; ++j) {
                        L_PLACEHOLDER[j][iiic] += l[j][ci];
                    }

                    if (opts.ang_con) {
                        if (type == RHO && opts.ang_con) {
                            for (integer j = 0; j != NDIM; ++j) {
                                L_c_PLACEHOLDER[j][iiic] += lc[j][ci];
                            }
                        }
                    }

                    if (!is_leaf) {
                        integer index = child_index(ip, jp, kp, ci, 0);
                        for (integer j = 0; j != 20; ++j) {
                            exp_ret.first[index][j] = L_PLACEHOLDER[j][iiic];
                        }

                        if (type == RHO && opts.ang_con) {
                            for (integer j = 0; j != 3; ++j) {
                                exp_ret.second[index][j] = L_c_PLACEHOLDER[j][iiic];
                            }
                        }
                    }
                }
            }
        }
    }

	if (is_leaf) {
		for (integer i = 0; i != G_NX; ++i) {
			for (integer j = 0; j != G_NX; ++j) {
				for (integer k = 0; k != G_NX; ++k) {
					const integer iii = gindex(i, j, k);
					const integer iii0 = h0index(i, j, k);
					const integer iiih = hindex(i + H_BW, j + H_BW, k + H_BW);
					if (type == RHO) {
						G[iii][phi_i] = physcon.G * L_PLACEHOLDER()[iii];
						for (integer d = 0; d < NDIM; ++d) {
							G[iii][gx_i + d] = -physcon.G * L_PLACEHOLDER(d)[iii];
							if (opts.ang_con == true) {
								G[iii][gx_i + d] -= physcon.G * L_c_PLACEHOLDER[d][iii];
							}
						}
						U[pot_i][iiih] = G[iii][phi_i] * U[rho_i][iiih];
					} else {
						dphi_dt[iii0] = physcon.G * L_PLACEHOLDER()[iii];
					}
				}
			}
		}
	} PROF_END;
    return exp_ret;
}

multipole_pass_type grid::compute_multipoles(
    gsolve_type type, const multipole_pass_type* child_poles) {
    //	if( int(*Muse_counter) > 0)
    //	printf( "%i\n", int(*Muse_counter));
//	printf( "%\n", scaling_factor);
	//if( dx  > 1.0e+12)
//	printf( "+-00000--------++++++++++++++++++++++++++++ %e \n", dx);

    PROF_BEGIN;

    integer lev = 0;
    const real dx3 = dx * dx * dx;
    M_ptr = std::make_shared<taylor<4, std::vector<real>>>();
    mon_ptr = std::make_shared<std::vector<real>>();
    if (com1_ptr == nullptr) { // FIXME: Should this only be allocated if type == RHO? [Bryce]
        com1_ptr = std::make_shared<std::array<std::array<real, G_N3 / 8>, NDIM>>();
        //for (std::array<real, G_N3 / 8>& a : *com1_ptr)
        //    std::fill(a.begin(), a.end(), ZERO);
    }
    if (type == RHO) { // FIXME: Should this only be allocated if com0_ptr != nullptr? [Bryce]
        com0_ptr = std::make_shared<std::array<std::array<real, G_N3>, NDIM>>();
        //for (std::array<real, G_N3>& a : *com0_ptr)
        //    std::fill(a.begin(), a.end(), ZERO);
    }
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;
    auto& com0 = *com0_ptr;
    auto& com1 = *com1_ptr;

    if (is_leaf) {
        for (integer j = 0; j != taylor_sizes[3]; ++j)
            M[j].clear();
        mon.resize(G_N3);
    } else {
        for (integer j = 0; j != taylor_sizes[3]; ++j)
            M[j].resize(G_N3);
        mon.resize(0);
    }
    if (type == RHO) {
        const integer iii0 = hindex(H_BW, H_BW, H_BW);
        const std::array<real, NDIM> x0 = {X[XDIM][iii0], X[YDIM][iii0], X[ZDIM][iii0]};
        for (integer i = 0; i != G_NX; ++i) {
            for (integer j = 0; j != G_NX; ++j) {
                for (integer k = 0; k != G_NX; ++k) {
                    integer const iii  = gindex(i, j, k); 
                    com0[0][iii] = x0[0] + i * dx;
                    com0[1][iii] = x0[1] + j * dx;
                    com0[2][iii] = x0[2] + k * dx;
                    //if (std::abs(com0[0][iii0]) > 1.0e+12)
                    //    printf( "!!!!!!!!!!!!!  %e %i %e %e  !!!!!!!!!!!!1111 \n", x0[0], int(i), dx, com0[0][iii0]);
                }
            }
        }
    }

    multipole_pass_type mret;
    if (!is_root) {
        mret.first.resize(INX * INX * INX / NCHILD);
        assert(com1[0].size() == INX * INX * INX / NCHILD); // TODO: REMOVE ME
        mret.second.resize(com1[0].size());
    }
    taylor<4, real> MM;
    integer index = 0;
    for (integer inx = INX; (inx >= INX / 2); inx >>= 1) {
        const integer nxp = inx;
        const integer nxc = (2 * inx);

        auto child_index = [=](integer ip, integer jp, integer kp, integer ci) -> integer {
            const integer ic = (2 * ip) + ((ci >> 0) & 1);
            const integer jc = (2 * jp) + ((ci >> 1) & 1);
            const integer kc = (2 * kp) + ((ci >> 2) & 1);
            return nxc * nxc * ic + nxc * jc + kc;
        };

        for (integer ip = 0; ip != nxp; ++ip) {
            for (integer jp = 0; jp != nxp; ++jp) {
                for (integer kp = 0; kp != nxp; ++kp) {
                    const integer iiip = nxp * nxp * ip + nxp * jp + kp;
                    if (lev != 0) {
                        if (type == RHO) {
                            simd_vector mc;
                            std::array<simd_vector, NDIM> X;
                            for (integer ci = 0; ci != NCHILD; ++ci) {
                                const integer iiic = child_index(ip, jp, kp, ci);
                                if (is_leaf) {
                                    mc[ci] = mon[iiic];
                                } else {
                                    mc[ci] = M()[iiic];
                                }
                                for (integer d = 0; d < NDIM; ++d) {
                                    X[d][ci] = com0[d][iiic];
                                }
                            }
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                            real mtot = mc.sum();
#else
                            real mtot = Vc::reduce(mc);
#endif
                            for (integer d = 0; d < NDIM; ++d) {
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                                com1[d][iiip] = (X[d] * mc).sum() / mtot;
#else
                                com1[d][iiip] = Vc::reduce(X[d] * mc) / mtot;
#endif
                            }
                        }
                        taylor<4, simd_vector> mc, mp;
                        std::array<simd_vector, NDIM> x, y, dx;
                        for (integer ci = 0; ci != NCHILD; ++ci) {
                            const integer iiic = child_index(ip, jp, kp, ci);
                            if (is_leaf) {
                                mc()[ci] = mon[iiic];
                                for (integer j = 1; j != 20; ++j) {
                                    mc.ptr()[j][ci] = 0.0;
                                }
                            } else {
                                for (integer j = 0; j != 20; ++j) {
                                    mc.ptr()[j][ci] = M[j][iiic];
                                }
                            }
                            for (integer d = 0; d < NDIM; ++d) {
                                x[d][ci] = com0[d][iiic];
                            }
                        }
                        for (integer d = 0; d < NDIM; ++d) {
                            dx[d] = x[d] - simd_vector(com1[d][iiip]);
                 //           if( std::abs(x[d][0]) > 5.0 )
                   //         printf( "%e %e ---\n", x[d][0], Y[d]);
                        }
                        mp = mc >> dx;
                        for (integer j = 0; j != 20; ++j) {
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                            MM[j] = mp[j].sum();
#else
                            MM[j] = Vc::reduce(mp[j]);
#endif
                        }
                    } else { // lev == 0
                        if (child_poles == nullptr) {
                            const integer iiih = hindex(ip + H_BW, jp + H_BW, kp + H_BW);
                            const integer iii0 = h0index(ip, jp, kp);
                            if (type == RHO) {
                                mon[iiip] = U[rho_i][iiih] * dx3;
                            } else {
                                mon[iiip] = dUdt[rho_i][iii0] * dx3;
                            }
                        } else {
                            for (integer j = 0; j != taylor_sizes[3]; ++j)
                                M[j][iiip] = child_poles->first[index][j];

                            if (type == RHO) {
                                for (integer d = 0; d != NDIM; ++d)
                                    com0[d][iiip] = child_poles->second[index][d];
                            }
                            ++index;
                        }
                    }
                    if (!is_root && (lev == 1)) {
                        mret.first[index] = MM;
                        for (integer d = 0; d != NDIM; ++d)
                            mret.second[index][d] = com1[d][iiip];
                        ++index;
                    }
                }
            }
        }
        ++lev;
        index = 0;
    }
    PROF_END;
    return mret;
}

gravity_boundary_type grid::get_gravity_boundary(const geo::direction& dir, bool is_local) {
    PROF_BEGIN;
    //	std::array<integer, NDIM> lb, ub;
    gravity_boundary_type data;
    data.is_local = is_local;
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

    if (!is_local) {
        data.allocate();
        integer iter = 0;
        const std::vector<boundary_interaction_type>& list = ilist_n_bnd[dir.flip()];
        if (is_leaf) {
            data.m->reserve(list.size());
            for (auto i : list) {
                const integer iii = i.second;
                data.m->push_back(mon[iii]);
            }
        } else {
            for (integer j = 0; j != taylor_sizes[3]; ++j)
                (*data.M)[j].reserve(list.size());
            //data.x->reserve(list.size());
            for (integer i = 0; i != list.size(); ++i) {
                const integer iii = list[i].second;
                for (integer j = 0; j != taylor_sizes[3]; ++j)
                    (*data.M)[j].push_back(M[j][iii]);
                for (integer d = 0; d != NDIM; ++d)
                {
                    (*data.com)[d][i] = (*com0_ptr)[d][iii];
                }
            }
        }
    } else {
        if (is_leaf) {
            data.m = mon_ptr;
        } else {
            data.M = M_ptr;
            data.com = com0_ptr;
        }
    }
    PROF_END;
    return data;
}
