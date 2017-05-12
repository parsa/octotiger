#include "m2m_interactions.hpp"

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {

    m2m_interactions::m2m_interactions(std::vector<multipole>& M_ptr,
        std::vector<std::shared_ptr<std::vector<space_vector>>> com_ptr,
        std::array<boundary_interaction_type, geo::direction::count()> neighbors)
      : verbose(true) {
        // allocate input variables with padding (so that the interactions spheres are always valid
        // indices)

        if (verbose) {
            std::cout << "EXPANSION_COUNT_PADDING: " << EXPANSION_COUNT_PADDING << std::endl;
            std::cout << "EXPANSION_COUNT_NOT_PADDED: " << EXPANSION_COUNT_NOT_PADDED << std::endl;
            std::cout << "EXPANSION_COUNT_PADDED: " << EXPANSION_COUNT_PADDED << std::endl;
        }
        local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
        center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

        std::vector<space_vector> const& com0 = *(com_ptr[0]);

        iterate_inner_cells_padded([this, M_ptr, com0](multiindex& i, size_t flat_index) {
            multiindex i_unpadded(i.x - INNER_CELLS_PADDING_DEPTH, i.y - INNER_CELLS_PADDING_DEPTH,
                i.z - INNER_CELLS_PADDING_DEPTH);
            size_t flat_index_unpadded = to_inner_flat_index_not_padded(i_unpadded);
            local_expansions[flat_index] = M_ptr[flat_index_unpadded];
            center_of_masses[flat_index] = com0[flat_index_unpadded];
        });

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
    }

    m2m_interactions::m2m_interactions()
      : verbose(true) {
        // allocate input variables with padding (so that the interactions spheres are always valid
        // indices)

        if (verbose) {
            std::cout << "EXPANSION_COUNT_PADDING: " << EXPANSION_COUNT_PADDING << std::endl;
            std::cout << "EXPANSION_COUNT_NOT_PADDED: " << EXPANSION_COUNT_NOT_PADDED << std::endl;
            std::cout << "EXPANSION_COUNT_PADDED: " << EXPANSION_COUNT_PADDED << std::endl;
        }
        local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
        center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

        iterate_inner_cells_padded([this](multiindex& i, size_t flat_index) {
            multiindex i_unpadded(i.x - INNER_CELLS_PADDING_DEPTH, i.y - INNER_CELLS_PADDING_DEPTH,
                i.z - INNER_CELLS_PADDING_DEPTH);
            std::cout << "i is " << i << " i_unpadded is " << i_unpadded << std::endl;
        });

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
    }

    void m2m_interactions::get_converted_local_expansions(std::vector<multipole>& M_ptr) {
        iterate_inner_cells_padded([this, M_ptr](multiindex& i, size_t flat_index) {
            multiindex i_unpadded(i.x - INNER_CELLS_PADDING_DEPTH, i.y - INNER_CELLS_PADDING_DEPTH,
                i.z - INNER_CELLS_PADDING_DEPTH);
            size_t flat_index_unpadded = to_inner_flat_index_not_padded(i_unpadded);
            M_ptr[flat_index_unpadded] = local_expansions[flat_index];
        });
    }

    void m2m_interactions::get_converted_potential_expansions(std::vector<expansion>& L) {
        iterate_inner_cells_not_padded([this, &L](multiindex& i, size_t flat_index) {
            L[flat_index] = potential_expansions[flat_index];
        });
    }

    void m2m_interactions::get_converted_angular_corrections(std::vector<space_vector>& L_c) {
        iterate_inner_cells_not_padded([this, &L_c](multiindex& i, size_t flat_index) {
            L_c[flat_index] = angular_corrections[flat_index];
        });
    }

}    // namespace fmm
}    // namespace octotiger
