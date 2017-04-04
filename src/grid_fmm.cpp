/*
 * grid_fmm.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: dmarce1
 */
#include "grid.hpp"
#include "grid_flattened_indices.hpp"
#include "options.hpp"
#include "physcon.hpp"
#include "profiler.hpp"
#include "simd.hpp"
#include "taylor.hpp"

#include <hpx/include/parallel_for_loop.hpp>

#include <cstddef>
#include <utility>

#ifdef USE_GRAV_PAR
const auto for_loop_policy = hpx::parallel::execution::par;
#else
const auto for_loop_policy = hpx::parallel::execution::seq;
#endif

// why not lists per cell? (David)
// is a grid actually a cell? (David)
// what is the semantics of a space_vector? (David)

// non-root and non-leaf node (David)
std::vector<interaction_type> ilist_n;
// monopole-monopole interactions? (David)
std::vector<interaction_type> ilist_d;
// interactions of root node (empty list? boundary stuff?) (David)
std::vector<interaction_type> ilist_r;
std::vector<std::vector<boundary_interaction_type>> ilist_d_bnd(geo::direction::count());
std::vector<std::vector<boundary_interaction_type>> ilist_n_bnd(geo::direction::count());
taylor<4, real> factor;
extern options opts;

template <class T>
void load_multipole(taylor<4, T>& m, space_vector& c, const gravity_boundary_type& data,
    integer iter, bool monopole) {
    if (monopole) {
        m = T(0.0);
        m = T((*(data.m))[iter]);
    } else {
        auto const& tmp1 = (*(data.M))[iter];
#pragma GCC ivdep
        for (int i = 0; i != 20; ++i) {
            m[i] = tmp1[i];
        }
        auto const& tmp2 = (*(data.x))[iter];
#pragma GCC ivdep
        for (integer d = 0; d != NDIM; ++d) {
            c[d] = tmp2[d];
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
    std::vector<space_vector> const& com0 = *(com_ptr[0]);
    for (integer i = 0; i != G_NX; ++i) {
        for (integer j = 0; j != G_NX; ++j) {
            for (integer k = 0; k != G_NX; ++k) {
                const integer iii1 = gindex(i, j, k);
                const integer iii0 = gindex(i + H_BW, j + H_BW, k + H_BW);
                space_vector const& com0iii1 = com0[iii1];
                multipole const& Miii1 = M[iii1];
                for (integer n = 0; n != NDIM; ++n) {
                    real mass;
                    if (is_leaf) {
                        mass = mon[iii1];
                    } else {
                        mass = Miii1();
                    }
                    this_com[n] += mass * com0iii1[n];
                    mtot += mass;
                    for (integer m = 0; m != NDIM; ++m) {
                        if (!is_leaf) {
                            quad_moment[n][m] += Miii1(n, m);
                        }
                        quad_moment[n][m] += mass * com0iii1[n] * com0iii1[m];
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

void grid::solve_gravity(gsolve_type type) {
    compute_multipoles(type);
    compute_interactions(type);
    compute_expansions(type);
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

void grid::compute_boundary_interactions_multipole_monopole(gsolve_type type,
    const std::vector<boundary_interaction_type>& ilist_n_bnd,
    const gravity_boundary_type& mpoles) {
    PROF_BEGIN;
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;

    std::vector<space_vector> const& com0 = *(com_ptr[0]);
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
                    space_vector const& com0iii0 = com0[iii0];
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][i] = com0iii0[d];
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
                //                             A0(a) += m0(c, b) * D(a, b, c) * (factor(b, c) *
                //                             HALF);
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
                    //                                     const auto tmp = D(a, b, c, d) *
                    //                                     (factor(b, c, d) * SIXTH);
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
                    expansion& Liii0 = L[iii0];
#pragma GCC ivdep
                    for (integer j = 0; j != 4; ++j) {
                        Liii0[j] += A0[j][i];
                    }
                    if (type == RHO && opts.ang_con) {
                        space_vector& L_ciii0 = L_c[iii0];
#pragma GCC ivdep
                        for (integer j = 0; j != NDIM; ++j) {
                            L_ciii0[j] += B0[j][i];
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

    std::vector<space_vector> const& com0 = *(com_ptr[0]);
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
                    space_vector const& com0iii0 = com0[iii0];
#pragma GCC ivdep
                    for (integer d = 0; d < NDIM; ++d) {
                        X[d][i] = com0iii0[d];
                    }
                    if (type == RHO) {
                        multipole const& Miii0 = M[iii0];
                        real const tmp = m0[i] / Miii0();
#pragma GCC ivdep
                        for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                            n0[j][i] = -Miii0[j] * tmp;
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
                    //                                     const auto tmp = D(a, b, c, d) *
                    //                                     (factor(b, c, d) * SIXTH);
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

                    expansion tmp;
#pragma GCC ivdep
                    for (integer j = 0; j != taylor_sizes[3]; ++j) {
                        tmp[j] = A0[j][i];
                    }
                    L[iii0] += tmp;

                    if (type == RHO && opts.ang_con) {
                        space_vector& L_ciii0 = L_c[iii0];
#pragma GCC ivdep
                        for (integer j = 0; j != NDIM; ++j) {
                            L_ciii0[j] += B0[j][i];
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
        for_loop_policy, 0, ilist_n_bnd.size(), [&mpoles, &ilist_n_bnd, &d0, this](std::size_t si) {

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
                expansion& Liii0 = L[iii0];
#pragma GCC ivdep
                for (integer i = 0; i != 4; ++i) {
                    Liii0[i] += tmp1[i];
                }
            }
        });
    PROF_END;
}

void compute_ilist() {
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
                                    // if (iii1 > iii0) {
                                    ilist_n0.push_back(np);
                                    // }
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
                                    // if (iii1 > iii0) {
                                    ilist_r0.push_back(np);
                                    // }
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

    {
        std::int32_t cur_index = ilist_r[0].first;
        std::int32_t cur_index_start = 0;
        for (std::int32_t li = 0; li != ilist_r.size(); ++li) {
            // std::cout << ilist_r[li].first << " ?? " << cur_index << std::endl;
            if (ilist_r[li].first != cur_index) {
                // std::cout << "range size: " << (li - cur_index_start) << std::endl;
                ilist_r[cur_index_start].inner_loop_stop = li;
                cur_index = ilist_r[li].first;
                cur_index_start = li;
            }
        }
        // make sure the last element is handled correctly as well
        ilist_r[cur_index_start].inner_loop_stop = ilist_r.size();
    }

    {
        std::int32_t cur_index = ilist_n[0].first;
        std::int32_t cur_index_start = 0;
        for (std::int32_t li = 0; li != ilist_n.size(); ++li) {
            // std::cout << ilist_n[li].first << " ?? " << cur_index << std::endl;
            if (ilist_n[li].first != cur_index) {
                // std::cout << "range size: " << (li - cur_index_start) << std::endl;
                ilist_n[cur_index_start].inner_loop_stop = li;
                cur_index = ilist_n[li].first;
                cur_index_start = li;
            }
        }
        // make sure the last element is handled correctly as well
        ilist_n[cur_index_start].inner_loop_stop = ilist_n.size();
    }

    {
        std::int32_t cur_index = ilist_d[0].first;
        std::int32_t cur_index_start = 0;
        for (std::int32_t li = 0; li != ilist_d.size(); ++li) {
            // std::cout << ilist_d[li].first << " ?? " << cur_index << std::endl;
            if (ilist_d[li].first != cur_index) {
                // std::cout << "range size: " << (li - cur_index_start) << std::endl;
                ilist_d[cur_index_start].inner_loop_stop = li;
                cur_index = ilist_d[li].first;
                cur_index_start = li;
            }
        }
        // make sure the last element is handled correctly as well
        ilist_d[cur_index_start].inner_loop_stop = ilist_d.size();
    }

    // // change for boundary
    // {
    //     std::int32_t cur_index = ilist_n_bnd[0].first;
    //     std::int32_t cur_index_start = 0;
    //     for (std::int32_t li = 0; li != ilist_n_bnd.size(); ++li) {
    //         // std::cout << ilist_d[li].first << " ?? " << cur_index << std::endl;
    //         if (ilist_n_bnd[li].first != cur_index) {
    //             // std::cout << "range size: " << (li - cur_index_start) << std::endl;
    //             ilist_n_bnd[cur_index_start].inner_loop_stop = li;
    //             cur_index = ilist_n_bnd[li].first;
    //             cur_index_start = li;
    //         }
    //     }
    //     // make sure the last element is handled correctly as well
    //     ilist_n_bnd[cur_index_start].inner_loop_stop = ilist_n_bnd.size();
    // }
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
                        X[d][ci] = (*(com_ptr[0]))[iiic][d];
                    }
                }
                const auto& Y = (*(com_ptr[1]))[iiip];
                for (integer d = 0; d < NDIM; ++d) {
                    dX[d] = X[d] - Y[d];
                }
                l <<= dX;
                for (integer ci = 0; ci != NCHILD; ++ci) {
                    const integer iiic = child_index(ip, jp, kp, ci);
                    expansion& Liiic = L[iiic];
                    for (integer j = 0; j != 20; ++j) {
                        Liiic[j] += l[j][ci];
                    }

                    space_vector& L_ciiic = L_c[iiic];
                    if (opts.ang_con) {
                        if (type == RHO && opts.ang_con) {
                            for (integer j = 0; j != NDIM; ++j) {
                                L_ciiic[j] += lc[j][ci];
                            }
                        }
                    }

                    if (!is_leaf) {
                        integer index = child_index(ip, jp, kp, ci, 0);
                        for (integer j = 0; j != 20; ++j) {
                            exp_ret.first[index][j] = Liiic[j];
                        }

                        if (type == RHO && opts.ang_con) {
                            for (integer j = 0; j != 3; ++j) {
                                exp_ret.second[index][j] = L_ciiic[j];
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
                        G[iii][phi_i] = physcon.G * L[iii]();
                        for (integer d = 0; d < NDIM; ++d) {
                            G[iii][gx_i + d] = -physcon.G * L[iii](d);
                            if (opts.ang_con == true) {
                                G[iii][gx_i + d] -= physcon.G * L_c[iii][d];
                            }
                        }
                        U[pot_i][iiih] = G[iii][phi_i] * U[rho_i][iiih];
                    } else {
                        dphi_dt[iii0] = physcon.G * L[iii]();
                    }
                }
            }
        }
    }
    PROF_END;
    return exp_ret;
}

multipole_pass_type grid::compute_multipoles(
    gsolve_type type, const multipole_pass_type* child_poles) {
    //	if( int(*Muse_counter) > 0)
    //	printf( "%i\n", int(*Muse_counter));
    //	printf( "%\n", scaling_factor);
    // if( dx  > 1.0e+12)
    //	printf( "+-00000--------++++++++++++++++++++++++++++ %e \n", dx);

    PROF_BEGIN;
    integer lev = 0;
    const real dx3 = dx * dx * dx;
    M_ptr = std::make_shared<std::vector<multipole>>();
    mon_ptr = std::make_shared<std::vector<real>>();
    if (com_ptr[1] == nullptr) {
        com_ptr[1] = std::make_shared<std::vector<space_vector>>(G_N3 / 8);
    }
    if (type == RHO) {
        com_ptr[0] = std::make_shared<std::vector<space_vector>>(G_N3);
    }
    auto& M = *M_ptr;
    auto& mon = *mon_ptr;
    if (is_leaf) {
        M.resize(0);
        mon.resize(G_N3);
    } else {
        M.resize(G_N3);
        mon.resize(0);
    }
    if (type == RHO) {
        const integer iii0 = hindex(H_BW, H_BW, H_BW);
        const std::array<real, NDIM> x0 = {X[XDIM][iii0], X[YDIM][iii0], X[ZDIM][iii0]};
        for (integer i = 0; i != G_NX; ++i) {
            for (integer j = 0; j != G_NX; ++j) {
                for (integer k = 0; k != G_NX; ++k) {
                    auto& com0iii = (*(com_ptr[0]))[gindex(i, j, k)];
                    com0iii[0] = x0[0] + i * dx;
                    com0iii[1] = x0[1] + j * dx;
                    com0iii[2] = x0[2] + k * dx;
                    //     if( std::abs(com0iii[0]) > 1.0e+12)
                    //      printf( "!!!!!!!!!!!!!  %e  %i %e %e  !!!!!!!!!!!!1111 \n", x0[0],
                    //      int(i), dx, com0iii[0]);
                }
            }
        }
    }

    multipole_pass_type mret;
    if (!is_root) {
        mret.first.resize(INX * INX * INX / NCHILD);
        mret.second.resize(INX * INX * INX / NCHILD);
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
                                    mc[ci] = M[iiic]();
                                }
                                for (integer d = 0; d < NDIM; ++d) {
                                    X[d][ci] = (*(com_ptr[0]))[iiic][d];
                                }
                            }
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                            real mtot = mc.sum();
#else
                            real mtot = Vc::reduce(mc);
#endif
                            for (integer d = 0; d < NDIM; ++d) {
#if !defined(HPX_HAVE_DATAPAR_VC) || (defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1)
                                (*(com_ptr[1]))[iiip][d] = (X[d] * mc).sum() / mtot;
#else
                                (*(com_ptr[1]))[iiip][d] = Vc::reduce(X[d] * mc) / mtot;
#endif
                            }
                        }
                        taylor<4, simd_vector> mc, mp;
                        std::array<simd_vector, NDIM> x, y, dx;
                        for (integer ci = 0; ci != NCHILD; ++ci) {
                            const integer iiic = child_index(ip, jp, kp, ci);
                            const space_vector& X = (*(com_ptr[lev - 1]))[iiic];
                            if (is_leaf) {
                                mc()[ci] = mon[iiic];
                                for (integer j = 1; j != 20; ++j) {
                                    mc.ptr()[j][ci] = 0.0;
                                }
                            } else {
                                for (integer j = 0; j != 20; ++j) {
                                    mc.ptr()[j][ci] = M[iiic].ptr()[j];
                                }
                            }
                            for (integer d = 0; d < NDIM; ++d) {
                                x[d][ci] = X[d];
                            }
                        }
                        const space_vector& Y = (*(com_ptr[lev]))[iiip];
                        for (integer d = 0; d < NDIM; ++d) {
                            dx[d] = x[d] - simd_vector(Y[d]);
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
                    } else {
                        if (child_poles == nullptr) {
                            const integer iiih = hindex(ip + H_BW, jp + H_BW, kp + H_BW);
                            const integer iii0 = h0index(ip, jp, kp);
                            if (type == RHO) {
                                mon[iiip] = U[rho_i][iiih] * dx3;
                            } else {
                                mon[iiip] = dUdt[rho_i][iii0] * dx3;
                            }
                        } else {
                            M[iiip] = child_poles->first[index];
                            if (type == RHO) {
                                (*(com_ptr)[lev])[iiip] = child_poles->second[index];
                            }
                            ++index;
                        }
                    }
                    if (!is_root && (lev == 1)) {
                        mret.first[index] = MM;
                        mret.second[index] = (*(com_ptr[lev]))[iiip];
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
            data.M->reserve(list.size());
            data.x->reserve(list.size());
            for (auto i : list) {
                const integer iii = i.second;
                const integer top = M[iii].size();
                data.M->push_back(M[iii]);
                data.x->push_back((*(com_ptr[0]))[iii]);
            }
        }
    } else {
        if (is_leaf) {
            data.m = mon_ptr;
        } else {
            data.M = M_ptr;
            data.x = com_ptr[0];
        }
    }
    PROF_END;
    return data;
}
