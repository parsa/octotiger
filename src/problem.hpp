/*
 * problem.hpp
 *
 *  Created on: May 29, 2015
 *      Author: dmarce1
 */

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include "defs.hpp"
#include <vector>
#include <functional>

using init_func_type = std::function<std::vector<real>(real,real,real,real)>;
using analytic_func_type = init_func_type;
using refine_test_type = std::function<bool(integer,integer, real,real,real,std::vector<real> const&,std::array<std::vector<real>,NDIM> const&)>;

OCTOTIGER_EXPORT std::vector<real> old_scf(real, real, real, real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> blast_wave(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> sod_shock_tube(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> star(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> moving_star_analytic(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> moving_star(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> equal_mass_binary(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> scf_binary(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> null_problem(real x, real y, real z, real);
OCTOTIGER_EXPORT std::vector<real> solid_sphere(real, real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> solid_sphere_analytic_phi(real x, real y, real z, real);
OCTOTIGER_EXPORT std::vector<real> double_solid_sphere(real, real, real, real);
OCTOTIGER_EXPORT std::vector<real> double_solid_sphere_analytic_phi(real x, real y, real z);

OCTOTIGER_EXPORT bool refine_test(integer level, integer maxl, real,real,real, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_test_bibi(integer level, integer maxl, real,real,real, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_sod(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);
OCTOTIGER_EXPORT bool refine_blast(integer level, integer max_level, real x, real y, real z, std::vector<real> const& U, std::array<std::vector<real>, NDIM> const& dudx);

OCTOTIGER_EXPORT void set_refine_test(const refine_test_type&);
OCTOTIGER_EXPORT refine_test_type get_refine_test();
OCTOTIGER_EXPORT void set_problem(const init_func_type&);
OCTOTIGER_EXPORT init_func_type get_problem();

#endif /* PROBLEM_HPP_ */
