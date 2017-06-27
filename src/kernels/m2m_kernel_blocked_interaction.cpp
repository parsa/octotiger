#include "m2m_kernel.hpp"

#include "grid_flattened_indices.hpp"
#include "helper.hpp"
#include "struct_of_array_taylor.hpp"
#include "m2m_taylor_set_basis.hpp"

extern taylor<4, real> factor;

namespace octotiger {
namespace fmm {

    //TODO:
    // - check codegen and fix in Vc
    // - check for amount of temporaries
    // - try to replace expensive operations like sqrt
    // - remove all sqr()
    // - increase INX

    void m2m_kernel::blocked_interaction(const multiindex<>& cell_index,
        const int64_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
        const multiindex<>& cell_index_unpadded, const int64_t cell_flat_index_unpadded,
        const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index) {
        // TODO: should change name to something better (not taylor, but space_vector)
        struct_of_array_taylor<space_vector, real, 3> X =
            center_of_masses_SoA.get_view(cell_flat_index);
        for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
             outer_stencil_index + inner_stencil_index < stencil.size();
             inner_stencil_index += 1) {
            const multiindex<>& stencil_element =
                stencil[outer_stencil_index + inner_stencil_index];
            const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

            // std::cout << " interaction_partner_index x: " << interaction_partner_index.x
            //           << " y: " << interaction_partner_index.y
            //           << " z: " << interaction_partner_index.z << std::endl;

            // check whether all vector elements are in empty border
            const multiindex<> in_boundary_start(
                (interaction_partner_index.x / INNER_CELLS_PER_DIRECTION) - 1,
                (interaction_partner_index.y / INNER_CELLS_PER_DIRECTION) - 1,
                (interaction_partner_index.z / INNER_CELLS_PER_DIRECTION) - 1);

#if Vc_IS_VERSION_2 == 0
            const multiindex<> in_boundary_end(
                (interaction_partner_index.x / INNER_CELLS_PER_DIRECTION) - 1,
                (interaction_partner_index.y / INNER_CELLS_PER_DIRECTION) - 1,
                ((interaction_partner_index.z + m2m_int_vector::Size) /
                    INNER_CELLS_PER_DIRECTION) -
                    1);
#else
            const multiindex<> in_boundary_end(
                (interaction_partner_index.x / INNER_CELLS_PER_DIRECTION) - 1,
                (interaction_partner_index.y / INNER_CELLS_PER_DIRECTION) - 1,
                ((interaction_partner_index.z + m2m_int_vector::size()) /
                    INNER_CELLS_PER_DIRECTION) -
                    1);
#endif

            // std::cout << "in_boundary_start x: " << in_boundary_start.x
            //           << " y: " << in_boundary_start.y << " z: " << in_boundary_start.z
            //           << std::endl;

            // std::cout << "in_boundary_end x: " << in_boundary_end.x << " y: " <<
            // in_boundary_end.y
            //           << " z: " << in_boundary_end.z << std::endl;

            geo::direction dir_start;
            dir_start.set(in_boundary_start.x, in_boundary_start.y, in_boundary_start.z);

            // std::cout << "dir_start flat_index: " << dir_start.flat_index_with_center()
            //           << std::endl;

            geo::direction dir_end;
            dir_end.set(in_boundary_end.x, in_boundary_end.y, in_boundary_end.z);

            // std::cout << "dir_end flat_index: " << dir_end.flat_index_with_center() << std::endl;

            if (neighbor_empty[dir_start.flat_index_with_center()] &&
                neighbor_empty[dir_end.flat_index_with_center()]) {
                continue;
            }

            const int64_t interaction_partner_flat_index =
                to_flat_index_padded(interaction_partner_index);

            // implicitly broadcasts to vector
            multiindex<m2m_int_vector> interaction_partner_index_coarse(interaction_partner_index);

#if Vc_IS_VERSION_2 == 0
            for (size_t j = 0; j < m2m_int_vector::Size; j++) {
#else
            for (size_t j = 0; j < m2m_int_vector::size(); j++) {
#endif
                interaction_partner_index_coarse.z[j] += j;
            }
            // note that this is the same for groups of 2x2x2 elements
            // -> maps to the same for some SIMD lanes
            interaction_partner_index_coarse.transform_coarse();

            m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                cell_index_coarse, interaction_partner_index_coarse);

            // m2m_vector theta_c_rec_squared =
            //     Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
            // TODO: use static_datapar_cast after issue is fixed
            m2m_vector theta_c_rec_squared;
            for (size_t i = 0; i < m2m_vector::size(); i++) {
                theta_c_rec_squared[i] = static_cast<double>(theta_c_rec_squared_int[i]);
            }

            m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

            if (Vc::none_of(mask)) {
                continue;
            }

            struct_of_array_taylor<space_vector, real, 3> Y =
                center_of_masses_SoA.get_view(interaction_partner_flat_index);

            // distance between cells in all dimensions
            // TODO: replace by m2m_vector for vectorization or get rid of temporary
            std::array<m2m_vector, NDIM> dX;
            for (integer d = 0; d < NDIM; ++d) {
                dX[d] = X.component(d) - Y.component(d);
            }

            struct_of_array_taylor<expansion, real, 20> m_partner =
                local_expansions_SoA.get_view(interaction_partner_flat_index);

            // TODO: replace by reference to get rid of temporary?
            taylor<4, m2m_vector> n0;

            // Initalize moments and momentum
            if (type != RHO) {
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = ZERO;
                }
            } else {
                // this branch computes the angular momentum correction, (20) in the
                // paper divide by mass of other cell
                struct_of_array_taylor<expansion, real, 20> m_cell =
                    local_expansions_SoA.get_view(cell_flat_index);

                m2m_vector const tmp1 = m_partner.grad_0() / m_cell.grad_0();

                // calculating the coefficients for formula (M are the octopole moments)
                // the coefficients are calculated in (17) and (18)
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = m_partner.component(j) - m_cell.component(j) * tmp1;
                }
            }

            // R_i in paper is the dX in the code
            // D is taylor expansion value for a given X expansion of the gravitational
            // potential
            // (multipole expansion)

            // calculates all D-values, calculate all coefficients of 1/r (not the
            // potential),
            // formula (6)-(9) and (19)
            // D.set_basis(dX);
            taylor<5, m2m_vector> D = set_basis(dX);

            // output variable references
            // TODO: use these again after debugging individual components!
            // expansion& current_potential =
            // potential_expansions[cell_flat_index_unpadded];
            // space_vector& current_angular_correction =
            // angular_corrections[cell_flat_index_unpadded];
            expansion_v current_potential;
            for (size_t i = 0; i < current_potential.size(); i++) {
                current_potential[i] = 0.0;
            }

            std::array<m2m_vector, NDIM> current_angular_correction;
            for (size_t i = 0; i < current_angular_correction.size(); i++) {
                current_angular_correction[i] = 0.0;
            }

            // the following loops calculate formula (10), potential from B->A
            current_potential[0] += m_partner.component(0) * D[0];
            if (type != RHO) {
                for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
                    current_potential[0] -= m_partner.component(i) * D[i];
                }
            }
            for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
                const auto tmp = D[i] * (factor[i] * HALF);
                current_potential[0] += m_partner.component(i) * tmp;
            }
            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                const auto tmp = D[i] * (factor[i] * SIXTH);
                current_potential[0] -= m_partner.component(i) * tmp;
            }

            for (integer a = 0; a < NDIM; ++a) {
                int const* ab_idx_map = to_ab_idx_map3[a];
                int const* abc_idx_map = to_abc_idx_map3[a];

                current_potential(a) += m_partner.grad_0() * D(a);
                for (integer i = 0; i != 6; ++i) {
                    if (type != RHO && i < 3) {
                        current_potential(a) -= m_partner.grad_1(a) * D[ab_idx_map[i]];
                    }
                    const integer cb_idx = cb_idx_map[i];
                    const auto tmp1 = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                    current_potential(a) += m_partner.component(cb_idx) * tmp1;
                }
            }

            if (type == RHO) {
                for (integer a = 0; a != NDIM; ++a) {
                    int const* abcd_idx_map = to_abcd_idx_map3[a];
                    for (integer i = 0; i != 10; ++i) {
                        const integer bcd_idx = bcd_idx_map[i];
                        const m2m_vector tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                        current_angular_correction[a] -= n0[bcd_idx] * tmp;
                    }
                }
            }

            for (integer i = 0; i != 6; ++i) {
                int const* abc_idx_map6 = to_abc_idx_map6[i];

                integer const ab_idx = ab_idx_map6[i];
                current_potential[ab_idx] += m_partner.grad_0() * D[ab_idx];
                for (integer c = 0; c < NDIM; ++c) {
                    const auto& tmp = D[abc_idx_map6[c]];
                    current_potential[ab_idx] -= m_partner.grad_1(c) * tmp;
                }
            }

            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                const auto& tmp = D[i];
                current_potential[i] += m_partner.grad_0() * tmp;
            }

            // TODO: remove this when switching back to non-copy (reference-based)
            // approach
            struct_of_array_taylor<expansion, real, 20> current_potential_result =
                potential_expansions_SoA.get_view(cell_flat_index_unpadded);
            struct_of_array_taylor<space_vector, real, 3> current_angular_correction_result =
                angular_corrections_SoA.get_view(cell_flat_index_unpadded);

            for (size_t i = 0; i < current_potential.size(); i++) {
                m2m_vector tmp = current_potential_result.component(i) + current_potential[i];
#if Vc_IS_VERSION_2 == 0
                tmp.store(current_potential_result.component_pointer(i), mask);
#else
                Vc::where(mask, tmp).memstore(
                    current_potential_result.component_pointer(i), Vc::flags::element_aligned);
#endif
            }
            for (size_t i = 0; i < current_angular_correction.size(); i++) {
                m2m_vector tmp =
                    current_angular_correction_result.component(i) + current_angular_correction[i];
#if Vc_IS_VERSION_2 == 0
                tmp.store(current_angular_correction_result.component_pointer(i), mask);
#else
                Vc::where(mask, tmp).memstore(
                    current_angular_correction_result.component_pointer(i),
                    Vc::flags::element_aligned);
#endif
            }
        }
    }
}    // namespace fmm
}    // namespace octotiger
