#include "grid.hpp"
#include "options.hpp"
#include "physcon.hpp"
#include "profiler.hpp"
#include "simd.hpp"
#include "taylor.hpp"

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

void grid::compute_interactions_leaf(gsolve_type type) {

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
