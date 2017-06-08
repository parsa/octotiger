#include "m2m_kernel.hpp"

#include "defs.hpp"
#include "grid_flattened_indices.hpp"
#include "helper.hpp"
#include "interaction_types.hpp"
#include "options.hpp"
#include "struct_of_array_taylor.hpp"

#include <array>
#include <functional>

extern taylor<4, real> factor;

// std::vector<interaction_type> ilist_debugging;

extern options opts;

namespace octotiger {
namespace fmm {

    void m2m_kernel::single_interaction(const multiindex<>& cell_index,
        const int64_t cell_flat_index, const multiindex<>& cell_index_unpadded,
        const int64_t cell_flat_index_unpadded, const multiindex<>& interaction_partner_index,
        const int64_t interaction_partner_flat_index) {
        // const real theta0 = opts.theta;
        const simd_vector theta_rec_sqared = sqr(1.0 / opts.theta);

        // TODO: should change name to something better (not taylor, but space_vector)
        struct_of_array_taylor<space_vector, real, 3> X =
            center_of_masses_SoA.get_view(cell_flat_index);

        struct_of_array_taylor<space_vector, real, 3> Y =
            center_of_masses_SoA.get_view(interaction_partner_flat_index);

        // distance between cells in all dimensions
        // TODO: replace by simd_vector for vectorization or get rid of temporary
        std::array<simd_vector, NDIM> dX;
        for (integer d = 0; d < NDIM; ++d) {
            dX[d] = X.component(d) - Y.component(d);
        }

        struct_of_array_taylor<expansion, real, 20> m_partner =
            local_expansions_SoA.get_view(interaction_partner_flat_index);

        // TODO: replace by reference to get rid of temporary?
        taylor<4, simd_vector> n0;

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

            simd_vector const tmp1 = m_partner.grad_0() / m_cell.grad_0();

            // calculating the coefficients for formula (M are the octopole moments)
            // the coefficients are calculated in (17) and (18)
            for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                n0[j] = m_partner.component(j) - m_cell.component(j) * tmp1;
            }
        }

        // R_i in paper is the dX in the code
        // D is taylor expansion value for a given X expansion of the gravitational potential
        // (multipole expansion)
        taylor<5, simd_vector> D;    // TODO: replace by simd_vector for vectorization

        // calculates all D-values, calculate all coefficients of 1/r (not the potential),
        // formula (6)-(9) and (19)
        D.set_basis(dX);

        // output variable references
        // TODO: use these again after debugging individual components!
        // expansion& current_potential = potential_expansions[cell_flat_index_unpadded];
        // space_vector& current_angular_correction = angular_corrections[cell_flat_index_unpadded];
        expansion_v current_potential;
        for (size_t i = 0; i < current_potential.size(); i++) {
            current_potential[i] = 0.0;
        }

        std::array<simd_vector, NDIM> current_angular_correction;
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
                    const simd_vector tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
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

        // TODO: remove this when switching back to non-copy (reference-based) approach
        struct_of_array_taylor<expansion, real, 20> current_potential_result =
            potential_expansions_SoA.get_view(cell_flat_index_unpadded);
        struct_of_array_taylor<space_vector, real, 3> current_angular_correction_result =
            angular_corrections_SoA.get_view(cell_flat_index_unpadded);

        // indices on coarser level (for outer stencil boundary)
        // implicitly broadcasts to vector
        multiindex<int_simd_vector> cell_index_coarse(cell_index);
        for (size_t j = 0; j < int_simd_vector::Size; j++) {
            cell_index_coarse.z[j] += j;
        }
        cell_index_coarse.transform_coarse();

        // implicitly broadcasts to vector
        multiindex<int_simd_vector> interaction_partner_index_coarse(interaction_partner_index);
        for (size_t j = 0; j < int_simd_vector::Size; j++) {
            interaction_partner_index_coarse.z[j] += j;
        }
        interaction_partner_index_coarse.transform_coarse();

        const int_simd_vector theta_f_rec_sqared =
            detail::distance_squared(cell_index, interaction_partner_index);

        const int_simd_vector theta_c_rec_sqared =
            detail::distance_squared(cell_index_coarse, interaction_partner_index_coarse);

        simd_vector::mask_type mask =
            theta_rec_sqared > theta_c_rec_sqared && theta_rec_sqared <= theta_f_rec_sqared;

        for (size_t i = 0; i < current_potential.size(); i++) {
            simd_vector tmp = current_potential_result.component(i) + current_potential[i];
            tmp.store(current_potential_result.component_pointer(i), mask);
        }
        for (size_t i = 0; i < current_angular_correction.size(); i++) {
            simd_vector tmp =
                current_angular_correction_result.component(i) + current_angular_correction[i];
            tmp.store(current_angular_correction_result.component_pointer(i), mask);
        }
    }

    m2m_kernel::m2m_kernel(struct_of_array_data<expansion, real, 20>& local_expansions_SoA,
        struct_of_array_data<space_vector, real, 3>& center_of_masses_SoA,
        struct_of_array_data<expansion, real, 20>& potential_expansions_SoA,
        struct_of_array_data<space_vector, real, 3>& angular_corrections_SoA, gsolve_type type)
      :    // local_expansions(local_expansions),
      local_expansions_SoA(local_expansions_SoA)
      // , center_of_masses(center_of_masses)
      , center_of_masses_SoA(center_of_masses_SoA)
      // , potential_expansions(potential_expansions)
      , potential_expansions_SoA(potential_expansions_SoA)
      // , angular_corrections(angular_corrections)
      , angular_corrections_SoA(angular_corrections_SoA)
      , type(type) {
        std::cout << "kernel created" << std::endl;
    }

    void m2m_kernel::apply_stencil(std::vector<multiindex<>>& stencil) {
        // for (multiindex<>& stencil_element : stencil) {
        for (size_t outer_stencil_index = 0; outer_stencil_index < stencil.size();
             outer_stencil_index += STENCIL_BLOCKING) {
            // std::cout << "stencil_element: " << stencil_element << std::endl;
            // TODO: remove after proper vectorization
            // multiindex<> se(stencil_element.x, stencil_element.y, stencil_element.z);
            // std::cout << "se: " << se << std::endl;
            // iterate_inner_cells_padded_stencil(se, *this);
            for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                    // for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                    for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2 += simd_vector::Size) {
                        const multiindex<> cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                            i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                        // BUG: indexing has to be done with uint32_t because of Vc limitation
                        const int64_t cell_flat_index = to_flat_index_padded(cell_index);
                        const multiindex<> cell_index_unpadded(i0, i1, i2);
                        const int64_t cell_flat_index_unpadded =
                            to_inner_flat_index_not_padded(cell_index_unpadded);

                        for (size_t inner_stencil_index = 0;
                             inner_stencil_index < STENCIL_BLOCKING &&
                             outer_stencil_index + inner_stencil_index < stencil.size();
                             inner_stencil_index += 1) {
                            const multiindex<>& stencil_element =
                                stencil[outer_stencil_index + inner_stencil_index];
                            const multiindex<> interaction_partner_index(
                                cell_index.x + stencil_element.x, cell_index.y + stencil_element.y,
                                cell_index.z + stencil_element.z);

                            const int64_t interaction_flat_partner_index =
                                to_flat_index_padded(interaction_partner_index);
                            this->single_interaction(cell_index, cell_flat_index,
                                cell_index_unpadded, cell_flat_index_unpadded,
                                interaction_partner_index, interaction_flat_partner_index);
                        }
                    }
                }
            }
        }
    }

    // void m2m_kernel::apply_stencil_element(multiindex& stencil_element) {
    //     // execute kernel for individual interactions for every inner cell
    //     // use this object as a functor for the iteration
    //     iterate_inner_cells_padded_stencil(stencil_element, *this);

    //     // std::cout << "potential_expansions.size(): " << potential_expansions.size() <<
    //     std::endl;
    //     // std::cout << "potential_expansions[0]: " << potential_expansions[0];
    //     // std::cout << std::endl;
    // }

}    // namespace fmm
}    // namespace octotiger
