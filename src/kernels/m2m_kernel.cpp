#include "m2m_kernel.hpp"

#include "grid_flattened_indices.hpp"

#include <array>
#include <functional>

extern taylor<4, real> factor;

namespace octotiger {
namespace fmm {

    namespace detail {

        void single_interaction(std::vector<expansion>& local_expansions,
            std::vector<space_vector>& center_of_masses,
            std::vector<expansion>& potential_expansions,
            std::vector<space_vector>& angular_corrections, gsolve_type type,
            const multiindex& cell_index, const size_t cell_flat_index,
            const multiindex& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const multiindex& interaction_partner_index,
            const size_t interaction_partner_flat_index) {
            std::array<real, NDIM>
                X;    // TODO: replace by space_vector for vectorization or get rid of temporary
            for (integer d = 0; d < NDIM; ++d) {
                // TODO: need SoA for this access if vectorized
                X[d] = center_of_masses[cell_flat_index][d];
            }

            std::array<real, NDIM>
                Y;    // TODO: replace by space_vector for vectorization or get rid of temporary
            for (integer d = 0; d < NDIM; ++d) {
                Y[d] = center_of_masses[interaction_partner_flat_index][d];
            }

            // cell specific taylor series coefficients
            // multipole const& Miii1 = M_ptr[iii1];
            // taylor<4, real> m0;    // TODO: replace by expansion type or get rid of temporary
            //                        // (replace by simd_vector for vectorization)
            // for (integer j = 0; j != taylor_sizes[3]; ++j) {
            //     m0[j] = local_expansions[interaction_partner_flat_index][j];
            // }
            expansion& m_partner = local_expansions[interaction_partner_flat_index];

            // n angular momentum of the cells
            // TODO: replace by expansion type or get rid of temporary
            // (replace by simd_vector for vectorization)
            taylor<4, real> n0;

            // Initalize moments and momentum
            if (type != RHO) {
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = ZERO;
                }
            } else {
                // this branch computes the angular momentum correction, (20) in the
                // paper divide by mass of other cell
                expansion& m_cell = local_expansions[cell_flat_index];

                real const tmp1 = m_partner() / m_cell();
                // calculating the coefficients for formula (M are the octopole moments)
                // the coefficients are calculated in (17) and (18)
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = m_partner[j] - m_cell[j] * tmp1;
                }
            }

            // taylor<4, simd_vector> A0;
            // std::array<simd_vector, NDIM> B0 = {
            //     {simd_vector(ZERO), simd_vector(ZERO), simd_vector(ZERO)}};

            // distance between cells in all dimensions
            // TODO: replace by simd_vector for vectorization or get rid of temporary
            std::array<real, NDIM> dX;
            for (integer d = 0; d < NDIM; ++d) {
                dX[d] = X[d] - Y[d];
            }

            // R_i in paper is the dX in the code
            // D is taylor expansion value for a given X expansion of the gravitational potential
            // (multipole expansion)
            taylor<5, real> D;    // TODO: replace by simd_vector for vectorization

            // calculates all D-values, calculate all coefficients of 1/r (not the potential),
            // formula (6)-(9) and (19)
            D.set_basis(dX);    // make sure the vectorized version is called

            // output variable references
            expansion& current_potential = potential_expansions[cell_flat_index_unpadded];
            space_vector& current_angular_correction =
                angular_corrections[cell_flat_index_unpadded];

            // the following loops calculate formula (10), potential from B->A
            current_potential[0] += m_partner[0] * D[0];
            if (type != RHO) {
                for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
                    current_potential[0] -= m_partner[i] * D[i];
                }
            }
            for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
                const auto tmp = D[i] * (factor[i] * HALF);
                current_potential[0] += m_partner[i] * tmp;
            }
            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                const auto tmp = D[i] * (factor[i] * SIXTH);
                current_potential[0] -= m_partner[i] * tmp;
            }

            for (integer a = 0; a < NDIM; ++a) {
                int const* ab_idx_map = to_ab_idx_map3[a];
                int const* abc_idx_map = to_abc_idx_map3[a];

                current_potential(a) += m_partner() * D(a);
                for (integer i = 0; i != 6; ++i) {
                    if (type != RHO && i < 3) {
                        current_potential(a) -= m_partner(a) * D[ab_idx_map[i]];
                    }
                    const integer cb_idx = cb_idx_map[i];
                    const auto tmp1 = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
                    current_potential(a) += m_partner[cb_idx] * tmp1;
                }
            }

            if (type == RHO) {
                for (integer a = 0; a != NDIM; ++a) {
                    int const* abcd_idx_map = to_abcd_idx_map3[a];
                    for (integer i = 0; i != 10; ++i) {
                        const integer bcd_idx = bcd_idx_map[i];
                        const auto tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                        current_angular_correction[a] -= n0[bcd_idx] * tmp;
                    }
                }
            }

            for (integer i = 0; i != 6; ++i) {
                int const* abc_idx_map6 = to_abc_idx_map6[i];

                integer const ab_idx = ab_idx_map6[i];
                current_potential[ab_idx] += m_partner() * D[ab_idx];
                for (integer c = 0; c < NDIM; ++c) {
                    const auto& tmp = D[abc_idx_map6[c]];
                    current_potential[ab_idx] -= m_partner(c) * tmp;
                }
            }

            for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
                const auto& tmp = D[i];
                current_potential[i] += m_partner[0] * tmp;
            }
        }
    }

    m2m_kernel::m2m_kernel(std::vector<expansion>& local_expansions,
        std::vector<space_vector>& center_of_masses, std::vector<expansion>& potential_expansions,
        std::vector<space_vector>& angular_corrections, gsolve_type type)
      : local_expansions(local_expansions)
      , center_of_masses(center_of_masses)
      , potential_expansions(potential_expansions)
      , angular_corrections(angular_corrections)
      , type(type) {}

    void m2m_kernel::apply_stencil_element(multiindex& stencil_element) {
        // execute kernel for individual interactions for every inner cell
        auto bound_single_interaction = std::bind(detail::single_interaction, local_expansions,
            center_of_masses, potential_expansions, angular_corrections, type,
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
            std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
        iterate_inner_cells_padded_stencil(stencil_element, bound_single_interaction);
    }

}    // namespace fmm
}    // namespace octotiger
