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
static std::vector<interaction_type> ilist_n;
// monopole-monopole interactions? (David)
static std::vector<interaction_type> ilist_d;
// interactions of root node (empty list? boundary stuff?) (David)
static std::vector<interaction_type> ilist_r;
static std::vector<std::vector<boundary_interaction_type>> ilist_d_bnd(geo::direction::count());
static std::vector<std::vector<boundary_interaction_type>> ilist_n_bnd(geo::direction::count());
static taylor<4, real> factor;
extern options opts;

void load_multipole(multipole& m, com_type& c, const gravity_boundary_type& data, integer iter, bool monopole) {
	if (monopole) {
		m = (0.0);
		m = ((*(data.m))[iter]);
	} else {
		auto const& tmp1 = (*(data.M))[iter];

		for (int i = 0; i != 20; ++i) {
			m[i] = tmp1[i];
		}
		auto const& tmp2 = (*(data.x))[iter];

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
	}PROF_END;
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
	std::vector<com_type> const& com0 = *(com_ptr[0]);
	for (integer i = 0; i != G_NX; ++i) {
		for (integer j = 0; j != G_NX; ++j) {
			for (integer k = 0; k != G_NX; ++k) {
				const integer iii1 = gindex(i, j, k);
				const integer iii0 = gindex(i + H_BW, j + H_BW, k + H_BW);
				com_type const& com0iii1 = com0[iii1];
				multipole const& Miii1 = M[iii1];
				for (integer n = 0; n != NDIM; ++n) {
					simd_vector mass;
					if (is_leaf) {
						mass = mon[iii1];
					} else {
						mass = Miii1();
					}
					for (auto c : geo::octant::full_set()) {
						this_com += mass[c] * com0iii1[n][c];
					}
					mtot += mass.sum();
					for (integer m = 0; m != NDIM; ++m) {
						if (!is_leaf) {
							for (auto c : geo::octant::full_set()) {
								quad_moment[n][m] += Miii1(n, m)[c];
							}
						}
						for (auto c : geo::octant::full_set()) {
							quad_moment[n][m] += mass[c] * com0iii1[n][c] * com0iii1[m][c];
						}
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
	std::pair<space_vector, space_vector> pair { rc, this_com };
	PROF_END;
	return pair;
}

void grid::solve_gravity(gsolve_type type) {
	compute_multipoles(type);
	compute_interactions(type);
	compute_expansions(type);
}

constexpr int to_ab_idx_map3[3][3] = { { 4, 5, 6 }, { 5, 7, 8 }, { 6, 8, 9 } };

constexpr int cb_idx_map[6] = { 4, 5, 6, 7, 8, 9, };

constexpr int to_abc_idx_map3[3][6] = { { 10, 11, 12, 13, 14, 15, }, { 11, 13, 14, 16, 17, 18, }, { 12, 14, 15, 17, 18, 19, } };

constexpr int to_abcd_idx_map3[3][10] = { { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 }, { 21, 23, 24, 26, 27, 28, 30, 31, 32, 33 }, { 22, 24, 25, 27, 28, 29, 31,
		32, 33, 34 } };

constexpr int bcd_idx_map[10] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };

constexpr int to_abc_idx_map6[6][3] = { { 10, 11, 12 }, { 11, 13, 14 }, { 12, 14, 15 }, { 13, 16, 17 }, { 14, 17, 18 }, { 15, 18, 19 } };

constexpr int ab_idx_map6[6] = { 4, 5, 6, 7, 8, 9 };

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
		std::vector<com_type> const& com0 = (*(com_ptr[0]));
		// Do 8 cell-cell interactions at the same time (simd.hpp) (Dominic)
		// This could lead to problems either on Haswell or on KNL as the number is hardcoded
		// (David)
		// It is unclear what the underlying vectorization library does, if simd_len > architectural
		// length (David)
		for (integer li = 0; li != list_size; ++li) {
			for (auto& tchild : geo::octant::full_set()) {
				// dX is distance between X and Y
				// X and Y are the two cells interacting
				// X and Y store the 3D center of masses (per simd element, SoA style)
				com_type dX;
				space_vector X;
				com_type Y;

				// m multipole moments of the cells
				taylor<4, simd_vector> m0;
				taylor<4, real> m1;
				// n angular momentum of the cells
				taylor<4, simd_vector> n0;
				taylor<4, simd_vector> n1;

				// vector of multipoles (== taylor<4>s)
				// here the multipoles are used as a storage of expansion coefficients
				auto& M = *M_ptr;

				// FIXME: replace with vector-pack gather-loads
				// TODO: only uses first 10 coefficients? ask Dominic
				//	for (integer i = 0; i != simd_len && integer(li + i) < list_size; ++i) {
				const integer iii0 = this_ilist[li].first;
				const integer iii1 = this_ilist[li].second;
				com_type const& com0iii0 = com0[iii0];
				Y = com0[iii1];
				for (integer d = 0; d < NDIM; ++d) {
					// load the 3D center of masses
					X[d] = com0iii0[d][tchild];
				}

				// cell specific taylor series coefficients
				auto const& Miii0 = M[iii0];
				auto const& Miii1 = M[iii1];
				m0 = M[iii1];
				for (integer j = 0; j != taylor_sizes[3]; ++j) {
					m1[j] = Miii0[j][tchild];
				}

				// RHO computes the gravitational potential based on the mass density
				// non-RHO computes the time derivative (hydrodynamics code V_t)
				if (type == RHO) {
					// this branch computes the angular momentum correction, (20) in the paper
					// divide by mass of other cell
					simd_vector const tmp1 = Miii1() / Miii0()[tchild];
					simd_vector const tmp2 = Miii0()[tchild] / Miii1();
					// calculating the coefficients for formula (M are the octopole moments)
					// the coefficients are calculated in (17) and (18)
					// TODO: Dominic
					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j] = Miii1[j] - Miii0[j][tchild] * tmp1;
						n1[j] = simd_vector(Miii0[j][tchild]) - Miii1[j] * tmp2;
					}
				}

				if (type != RHO) {

					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j] = ZERO;
						n1[j] = ZERO;
					}
				}

// distance between cells in all dimensions

				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = simd_vector(X[d]) - Y[d];
				}

				// R_i in paper is the dX in the code
				// D is taylor expansion value for a given X expansion of the gravitational
				// potential (multipole expansion)
				taylor<5, simd_vector> D;
				// A0, A1 are the contributions to L (for each cell)
				taylor<4, simd_vector> A0, A1;
				// B0, B1 are the contributions to L_c (for each cell)
				std::array<simd_vector, NDIM> B0 = { simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO) };
				std::array<simd_vector, NDIM> B1 = { simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO) };

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

				// potential and correction have been computed, no scatter the results

				for (integer j = 0; j != taylor_sizes[3]; ++j) {
					L[iii0][j][tchild] += (A0[j] * this_ilist[li].mask[tchild]).sum();
				}
				L[iii1] += A1 * this_ilist[li].mask[tchild];

				if (type == RHO && opts.ang_con) {
					com_type& L_ciii0 = L_c[iii0];
					com_type& L_ciii1 = L_c[iii1];
					for (integer j = 0; j != NDIM; ++j) {
						L_ciii0[j][tchild] += (B0[j] * this_ilist[li].mask[tchild]).sum();
						L_ciii1[j] += B1[j] * this_ilist[li].mask[tchild];
					}
				}
			}
		}
	} else {

		auto& M = *M_ptr;
		auto& mon = *mon_ptr;

		std::array<real, NDIM> Xbase = { X[0][hindex(H_BW, H_BW, H_BW)], X[1][hindex(H_BW, H_BW, H_BW)], X[2][hindex(H_BW, H_BW, H_BW)] };

		std::vector<com_type> const& com0 = *(com_ptr[0]);
		for (integer li = 0; li != ilist_d.size(); ++li) {

			real m0;
			simd_vector m1;
			std::array<simd_vector, NDIM> dX;
			std::array < simd_vector, NDIM > X;
			space_vector Y;

			integer iii1 = ilist_d[li].second;
			const integer iii0 = ilist_d[li].first;
			for (auto cy : geo::octant::full_set()) {

				for (integer d = 0; d != NDIM; ++d) {
					Y[d] = ilist_d[li].x[d][cy] * dx + Xbase[d];
				}

				m0 = (*(mon_ptr))[iii1][cy];
				m1 = (*(mon_ptr))[iii1];
				com_type const& com0iii0 = com0[iii0];

				for (integer d = 0; d < NDIM; ++d) {
					X[d] = com0iii0[d];
				}

				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = X[d] - Y[d];
				}
				simd_vector dX2(1.0e-40);
				for (integer d = 0; d != NDIM; ++d) {
					dX2 += dX[d] * dX[d];
				}
				taylor<2, simd_vector> A0;
				taylor<2, simd_vector> A1;
				A1() = A0() = -simd_vector(1.0) / sqrt(dX2);
				for (integer d = 0; d != NDIM; ++d) {
					A0(d) = -dX[d] * A0() / dX2;
					A1(d) = -A0(d);
				}
				A1() *= m0;
				A0() *= m1;
				for (integer d = 0; d != NDIM; ++d) {
					A1(d) *= m0;
					A0(d) *= m1;
				}

				L[iii0] += A0 * ilist_d[li].mask[cy];
				for (integer j = 0; j != taylor_sizes[3]; ++j) {
					L[iii1][j][cy] += (A1[j] * ilist_d[li].mask[cy]).sum();
				}
			}
		}
	}PROF_END;
}

void grid::compute_boundary_interactions(gsolve_type type, const geo::direction& dir, bool is_monopole, const gravity_boundary_type& mpoles) {
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

void grid::compute_boundary_interactions_multipole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
		const gravity_boundary_type& mpoles) {

	PROF_BEGIN;
	auto& M = *M_ptr;
	auto& mon = *mon_ptr;

	std::vector<com_type> const& com0 = *(com_ptr[0]);
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {
		boundary_interaction_type const& bnd = ilist_n_bnd[si];
		integer index = mpoles.is_local ? bnd.second : si;
		taylor<4, simd_vector> m0_c;
		com_type Y_c;
		load_multipole(m0_c, Y_c, mpoles, index, false);
		for (integer li = 0; li != bnd.first.size(); ++li) {
			for (auto cy : geo::octant::full_set()) {
				taylor<4, real> m0;
				space_vector Y;

				for (integer j = 0; j != taylor_sizes[3]; ++j) {
					m0[j] = m0_c[j][cy];
				}

				for (integer d = 0; d != NDIM; ++d) {
					Y[d] = Y_c[d][cy];
				}
				taylor<4, simd_vector> n0;
				com_type dX;
				com_type X;

				std::array<simd_vector, NDIM> simdY = { simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])), };

				const integer list_size = bnd.first.size();
				for (integer li = 0; li < list_size; li += simd_len) {
					const integer iii0 = bnd.first[li];
					com_type const& com0iii0 = com0[iii0];

					for (integer d = 0; d < NDIM; ++d) {
						X[d] = com0iii0[d];
					}
					if (type == RHO) {
						multipole const& Miii0 = M[iii0];
						const simd_vector tmp = m0() / Miii0();
						for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
							n0[j] = simd_vector(m0[j]) - Miii0[j] * tmp;
						}
					} else {

						for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
							n0[j] = ZERO;
						}
					}

					for (integer d = 0; d < NDIM; ++d) {
						dX[d] = X[d] - simdY[d];
					}

					taylor<5, simd_vector> D;
					taylor<4, simd_vector> A0;
					com_type B0;
					for (integer d = 0; d != NDIM; ++d) {
						B0[d] = simd_vector(0.0);
					}
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

					expansion& Liii0 = L[iii0];

					L[iii0] += A0 * bnd.mask[li][cy];

					if (type == RHO && opts.ang_con) {
						L_c[iii0] += B0 * bnd.mask[li][cy];
					}
				}
			}
		}
	}PROF_END;
}

void grid::compute_boundary_interactions_multipole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
		const gravity_boundary_type& mpoles) {
	PROF_BEGIN;
	auto& M = *M_ptr;
	auto& mon = *mon_ptr;
	std::vector<com_type> const& com0 = *(com_ptr[0]);
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {
		boundary_interaction_type const& bnd = ilist_n_bnd[si];
		integer index = mpoles.is_local ? bnd.second : si;
		taylor<4, simd_vector> m0_c;
		com_type Y_c;
		load_multipole(m0_c, Y_c, mpoles, index, false);
		for (auto cy : geo::octant::full_set()) {
			for (integer li = 0; li != bnd.first.size(); ++li) {
				taylor<4, real> m0;
				taylor<4, real> n0;
				com_type dX;
				space_vector Y;
				for (integer i = 0; i != taylor_sizes[3]; ++i) {
					m0[i] = m0_c[i][cy];
				}
				for (integer d = 0; d != NDIM; ++d) {
					Y[d] = Y_c[d][cy];
				}

				std::array<simd_vector, NDIM> simdY = { simd_vector(real(Y[0])), simd_vector(real(Y[1])), simd_vector(real(Y[2])), };

				if (type == RHO) {

					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j] = m0[j];
					}
				} else {

					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j] = ZERO;
					}
				}
				const integer iii0 = bnd.first[li];
				const auto X = com0[iii0];
				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = X[d] - simdY[d];
				}

				taylor<5, simd_vector> D;
				taylor<2, simd_vector> A0;
				com_type B0;
				for (integer d = 0; d != NDIM; ++d) {
					B0[d] = simd_vector(0.0);
				}
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
						A0(a) += m0[cb_idx] * D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
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

				L[iii0] += A0 * bnd.mask[li][cy];
				if (type == RHO && opts.ang_con) {
					L_c[iii0] += B0 * bnd.mask[li][cy];
				}
			}
		}
	};PROF_END;
}

void grid::compute_boundary_interactions_monopole_multipole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
		const gravity_boundary_type& mpoles) {

	PROF_BEGIN;
	auto& M = *M_ptr;
	auto& mon = *mon_ptr;

	std::array<real, NDIM> Xbase = { X[0][hindex(H_BW, H_BW, H_BW)], X[1][hindex(H_BW, H_BW, H_BW)], X[2][hindex(H_BW, H_BW, H_BW)] };

	std::vector<com_type> const& com0 = *(com_ptr[0]);
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {

		real m0;
		taylor<4, simd_vector> n0;
		std::array<simd_vector, NDIM> dX;
		std::array < simd_vector, NDIM > X;
		space_vector Y;

		boundary_interaction_type const& bnd = ilist_n_bnd[si];
		integer index = mpoles.is_local ? bnd.second : si;
		const integer list_size = bnd.first.size();
		for (auto cy : geo::octant::full_set()) {

			for (integer d = 0; d != NDIM; ++d) {
				Y[d] = bnd.x[d][cy] * dx + Xbase[d];
			}

			m0 = (*(mpoles.m))[index][cy];
			for (integer li = 0; li < list_size; li += simd_len) {
				const integer iii0 = bnd.first[li];
				com_type const& com0iii0 = com0[iii0];

				for (integer d = 0; d < NDIM; ++d) {
					X[d] = com0iii0[d];
				}
				if (type == RHO) {
					multipole const& Miii0 = M[iii0];
					simd_vector const tmp = m0 / Miii0();

					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j] = -Miii0[j] * tmp;
					}
				} else {

					for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
						n0[j] = ZERO;
					}
				}

				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = X[d] - Y[d];
				}

				taylor<5, simd_vector> D;
				taylor<4, simd_vector> A0;
				com_type B0;
				for (integer d = 0; d != NDIM; ++d) {
					B0[d] = simd_vector(0.0);
				}

				D.set_basis(dX);

				A0[0] = m0 * D[0];
				for (integer i = taylor_sizes[0]; i != taylor_sizes[2]; ++i) {
					A0[i] = m0 * D[i];
				}
				for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
					A0[i] = m0 * D[i];
				}

				if (type == RHO) {
					for (integer a = 0; a != NDIM; ++a) {
						int const* abcd_idx_map = to_abcd_idx_map3[a];
						for (integer i = 0; i != 10; ++i) {
							const integer bcd_idx = bcd_idx_map[i];
							const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
							B0[a] -= n0[bcd_idx] * tmp;
						}
					}
				}

				L[iii0] += A0 * bnd.mask[li][cy];
				if (type == RHO && opts.ang_con) {
					L_c[iii0] += B0 * bnd.mask[li][cy];
				}

			}
		}
	}PROF_END;
}

void grid::compute_boundary_interactions_monopole_monopole(gsolve_type type, const std::vector<boundary_interaction_type>& ilist_n_bnd,
		const gravity_boundary_type& mpoles) {

	PROF_BEGIN;
	auto& M = *M_ptr;
	auto& mon = *mon_ptr;

	std::array<real, NDIM> Xbase = { X[0][hindex(H_BW, H_BW, H_BW)], X[1][hindex(H_BW, H_BW, H_BW)], X[2][hindex(H_BW, H_BW, H_BW)] };

	std::vector<com_type> const& com0 = *(com_ptr[0]);
	for (integer si = 0; si != ilist_n_bnd.size(); ++si) {

		real m0;
		std::array<simd_vector, NDIM> dX;
		std::array < simd_vector, NDIM > X;
		space_vector Y;

		boundary_interaction_type const& bnd = ilist_n_bnd[si];
		integer index = mpoles.is_local ? bnd.second : si;
		const integer list_size = bnd.first.size();
		for (auto cy : geo::octant::full_set()) {

			for (integer d = 0; d != NDIM; ++d) {
				Y[d] = bnd.x[d][cy] * dx + Xbase[d];
			}

			m0 = (*(mpoles.m))[index][cy];
			for (integer li = 0; li < list_size; li += simd_len) {
				const integer iii0 = bnd.first[li];
				com_type const& com0iii0 = com0[iii0];

				for (integer d = 0; d < NDIM; ++d) {
					X[d] = com0iii0[d];
				}
				for (integer d = 0; d < NDIM; ++d) {
					dX[d] = X[d] - Y[d];
				}
				simd_vector dX2(1.0e-40);
				for (integer d = 0; d != NDIM; ++d) {
					dX2 += dX[d] * dX[d];
				}
				taylor<2, simd_vector> A0;
				A0() = -simd_vector(1.0) / sqrt(dX2) * m0;
				for (integer d = 0; d != NDIM; ++d) {
					A0(d) = -dX[d] * A0() / dX2 * m0;
				}
				L[iii0] += A0 * bnd.mask[li][cy];
			}
		}
	}PROF_END;
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
	for (integer i0_c = 0; i0_c != G_NX; ++i0_c) {
		for (integer j0_c = 0; j0_c != G_NX; ++j0_c) {
			for (integer k0_c = 0; k0_c != G_NX; ++k0_c) {
				integer this_d = 0;
				integer ilb = std::min(i0_c - G_NX, integer(0));
				integer jlb = std::min(j0_c - G_NX, integer(0));
				integer klb = std::min(k0_c - G_NX, integer(0));
				integer iub = std::max(i0_c + G_NX + 1, G_NX);
				integer jub = std::max(j0_c + G_NX + 1, G_NX);
				integer kub = std::max(k0_c + G_NX + 1, G_NX);
				for (integer i1_c = ilb; i1_c < iub; ++i1_c) {
					for (integer j1_c = jlb; j1_c < jub; ++j1_c) {
						for (integer k1_c = klb; k1_c < kub; ++k1_c) {
							const real theta_c = theta(i0_c, j0_c, k0_c, i1_c, j1_c, k1_c);
							const integer iii0 = gindex(i0_c, j0_c, k0_c);
							const integer iii1n = gindex((i1_c + G_NX) % G_NX, (j1_c + G_NX) % G_NX, (k1_c + G_NX) % G_NX);
							const integer iii1 = gindex(i1_c, j1_c, k1_c);
							if (theta_c > theta0) {
								++this_d;
								dp.first = iii0;
								dp.second = iii1n;
								np.first = iii0;
								np.second = iii1n;
								for (auto c1 : geo::octant::full_set()) {
									const integer i1_f = 2 * i1_c + c1[XDIM];
									const integer j1_f = 2 * j1_c + c1[YDIM];
									const integer k1_f = 2 * k1_c + c1[ZDIM];
									np.x[XDIM][c1] = i1_f;
									np.x[YDIM][c1] = j1_f;
									np.x[ZDIM][c1] = k1_f;
									dp.x[XDIM][c1] = i1_f;
									dp.x[YDIM][c1] = j1_f;
									dp.x[ZDIM][c1] = k1_f;
									for (auto c0 : geo::octant::full_set()) {
										const integer i0_f = 2 * i0_c + c0[XDIM];
										const integer j0_f = 2 * j0_c + c0[YDIM];
										const integer k0_f = 2 * k0_c + c0[ZDIM];
										const real theta_f = theta(i0_f, j0_f, k0_f, i1_f, j1_f, k1_f);
										if (theta_f >= theta0) {
											np.mask[c0][c1] = 0.0;
										} else {
											np.mask[c0][c1] = 1.0;
										}
										if (i1_c == i0_c && j0_c == j1_c && k0_c == k1_c) {
											if (i1_f == i0_f && j0_f == j1_f && k0_f == k1_f) {
												dp.mask[c0][c1] = 0.0;
											} else {
												dp.mask[c0][c1] = 0.5;
											}
										}
									}
								}
								if (interior(i1_c, j1_c, k1_c) && interior(i0_c, j0_c, k0_c)) {
									if (iii1 > iii0) {
										ilist_n0.push_back(np);
									}
								} else if (interior(i0_c, j0_c, k0_c)) {
									ilist_n0_bnd[neighbor_dir(i1_c, j1_c, k1_c)].push_back(np);
								}
								if (interior(i1_c, j1_c, k1_c) && interior(i0_c, j0_c, k0_c)) {
									if (iii1 > iii0) {
										ilist_d0.push_back(dp);
									}
								} else if (interior(i0_c, j0_c, k0_c)) {
									ilist_d0_bnd[neighbor_dir(i1_c, j1_c, k1_c)].push_back(dp);
								}
								if (interior(i1_c, j1_c, k1_c) && interior(i0_c, j0_c, k0_c)) {
									if (iii1 > iii0) {
										ilist_r0.push_back(np);
									}
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
					i.mask.push_back(i0.mask);
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
				i.mask.push_back(i0.mask);
				d.push_back(i);
			}
		}
		for (auto i0 : n0) {
			bool found = false;
			for (auto& i : n) {
				if (i.second == i0.second) {
					i.first.push_back(i0.first);
					i.mask.push_back(i0.mask);
					found = true;
					break;
				}
			}
			assert(found);
		}
	}
}

expansion_pass_type grid::compute_expansions(gsolve_type type, const expansion_pass_type* parent_expansions) {
	PROF_BEGIN;
	expansion_pass_type exp_ret;

	/*************************CONSTRUCTION ZONE*****************************

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
	 } PROF_END;
	 */
	return exp_ret;
}

multipole_pass_type grid::compute_multipoles(gsolve_type type, const multipole_pass_type* child_poles) {

	multipole_pass_type mret;
	PROF_BEGIN;
	integer lev = 0;
	const real dx3 = dx * dx * dx;
	M_ptr = std::make_shared<std::vector<multipole>>();
	mon_ptr = std::make_shared<std::vector<real>>();
	if (com_ptr[1] == nullptr) {
		com_ptr[1] = std::make_shared < std::vector < space_vector >> (G_N3 / 8);
	}
	if (type == RHO) {
		com_ptr[0] = std::make_shared < std::vector < space_vector >> (G_N3);
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
		const std::array<real, NDIM> x0 = { X[XDIM][iii0], X[YDIM][iii0], X[ZDIM][iii0] };
		for (integer i = 0; i != G_NX; ++i) {
			for (integer j = 0; j != G_NX; ++j) {
				for (integer k = 0; k != G_NX; ++k) {
					auto& com0iii = (*(com_ptr[0]))[gindex(i, j, k)];
					com0iii[0] = x0[0] + i * dx;
					com0iii[1] = x0[1] + j * dx;
					com0iii[2] = x0[2] + k * dx;
				}
			}
		}
	}
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
							std::array < simd_vector, NDIM > X;
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
							real mtot = mc.sum();
							for (integer d = 0; d < NDIM; ++d) {
								(*(com_ptr[1]))[iiip][d] = (X[d] * mc).sum() / mtot;
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
						}
						mp = mc >> dx;
						for (integer j = 0; j != 20; ++j) {
							MM[j] = mp[j].sum();
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
	} PROF_END;
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
	}PROF_END;
	return data;
}
