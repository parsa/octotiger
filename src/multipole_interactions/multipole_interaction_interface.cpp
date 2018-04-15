#include "multipole_interaction_interface.hpp"

#include "../common_kernel/interactions_iterators.hpp"
#include "calculate_stencil.hpp"
#include "multipole_cpu_kernel.hpp"
#include "options.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        thread_local const two_phase_stencil multipole_interaction_interface::stencil =
            calculate_stencil();
        thread_local std::vector<real>
            multipole_interaction_interface::local_monopoles_staging_area(EXPANSION_COUNT_PADDED);
        thread_local struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>
            multipole_interaction_interface::local_expansions_staging_area;
        thread_local struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>
            multipole_interaction_interface::center_of_masses_staging_area;
        multipole_interaction_interface::multipole_interaction_interface(void) {
            local_monopoles_staging_area = std::vector<real>(ENTRIES);
            this->m2m_type = opts.m2m_kernel_type;
        }

        void multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase) {
            update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                local_monopoles_staging_area, local_expansions_staging_area,
                center_of_masses_staging_area);
            compute_interactions(is_direction_empty, neighbors, local_monopoles_staging_area,
                local_expansions_staging_area, center_of_masses_staging_area);
        }

        void multipole_interaction_interface::compute_interactions_compute_block(
            const std::vector<real>& local_monopoles,
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA,
            size_t x, size_t y) {
            struct_of_array_data<expansion, real, 20, COMPUTE_BLOCK, SOA_PADDING>
                potential_expansions_SoA;
            struct_of_array_data<space_vector, real, 3, COMPUTE_BLOCK, SOA_PADDING>
                angular_corrections_SoA;

            multipole_cpu_kernel kernel;
            kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA,
                potential_expansions_SoA, angular_corrections_SoA, local_monopoles, stencil, type,
                x, y);

            std::vector<expansion>& L = grid_ptr->get_L();
            std::vector<space_vector>& L_c = grid_ptr->get_L_c();
            auto data_c = angular_corrections_SoA.get_pod();
            auto data = potential_expansions_SoA.get_pod();

            for (auto i0 = 0; i0 < COMPUTE_BLOCK_LENGTH; ++i0) {
                for (auto i1 = 0; i1 < COMPUTE_BLOCK_LENGTH; ++i1) {
                    for (auto i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; ++i2) {
                        const multiindex<> cell_index_unpadded(i0, i1, i2);
                        const int64_t cell_flat_index_unpadded =
                            i0 * COMPUTE_BLOCK_LENGTH * INNER_CELLS_PER_DIRECTION +
                            i1 * INNER_CELLS_PER_DIRECTION + i2;
                        const multiindex<> cell_index(i0 + x * COMPUTE_BLOCK_LENGTH,
                            i1 + y * COMPUTE_BLOCK_LENGTH, i2);
                        const int64_t cell_flat_index = to_inner_flat_index_not_padded(cell_index);

                        for (size_t component = 0; component < 20; component++) {
                            L[cell_flat_index][component] +=
                                data[component * (COMPUTE_BLOCK + SOA_PADDING) +
                                    cell_flat_index_unpadded];
                        }
                        for (size_t component = 0; component < 3 && type == RHO; component++) {
                            L_c[cell_flat_index][component] +=
                                data_c[component * (COMPUTE_BLOCK + SOA_PADDING) +
                                    cell_flat_index_unpadded];
                        }
                    }
                }
            }
        }

        void multipole_interaction_interface::compute_interactions(
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::vector<neighbor_gravity_type>& all_neighbor_interaction_data,
            const std::vector<real>& local_monopoles,
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA) {
            if (m2m_type == interaction_kernel_type::SOA_CPU) {
                for (auto x = 0; x < 2; ++x) {
                    for (auto y = 0; y < 2; ++y) {
                            compute_interactions_compute_block(local_monopoles,
                                local_expansions_SoA, center_of_masses_SoA, x, y);
                    }
                }
            } else {
                // old-style interaction calculation
                // computes inner interactions
                grid_ptr->compute_interactions(type);
                // computes boundary interactions
                for (auto const& dir : geo::direction::full_set()) {
                    if (!is_direction_empty[dir]) {
                        neighbor_gravity_type& neighbor_data = all_neighbor_interaction_data[dir];
                        grid_ptr->compute_boundary_interactions(type, neighbor_data.direction,
                            neighbor_data.is_monopole, neighbor_data.data);
                    }
                }
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
