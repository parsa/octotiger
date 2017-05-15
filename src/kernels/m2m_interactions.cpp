#include "m2m_interactions.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {

    m2m_interactions::m2m_interactions(
        grid& g, std::vector<node_server::neighbor_gravity_type>& neighbors)
      : verbose(true) {
        std::vector<multipole>& M_ptr = g.get_M();
        std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr = g.get_com_ptr();
        // allocate input variables with padding (so that the interactions spheres are always valid
        // indices)

        if (verbose) {
            std::cout << "EXPANSION_COUNT_NOT_PADDED: " << EXPANSION_COUNT_NOT_PADDED << std::endl;
            std::cout << "EXPANSION_COUNT_PADDED: " << EXPANSION_COUNT_PADDED << std::endl;
        }
        local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
        center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

        std::vector<bool> all_set(EXPANSION_COUNT_PADDED);
        std::fill(all_set.begin(), all_set.end(), false);

        std::vector<space_vector> const& com0 = *(com_ptr[0]);

        iterate_inner_cells_padded(
            [this, M_ptr, com0, &all_set](const multiindex& i, const size_t flat_index,
                const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                // std::cout << "flat_index: " << flat_index
                //           << " flat_index_unpadded: " << flat_index_unpadded << std::endl;
                local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
                all_set[flat_index] = true;
            });
        std::cout << "after iteration" << std::endl;

        for (const geo::direction& dir : geo::direction::full_set()) {
            std::cout << "dir 0: " << dir[0] << " 1: " << dir[1] << " 2: " << dir[2] << std::endl;
            iterate_inner_cells_padding(
                dir, [this, M_ptr, com0, &all_set](const multiindex& i, const size_t flat_index,
                         const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                    // std::cout << "flat_index: " << flat_index
                    //           << " flat_index_unpadded: " << flat_index_unpadded << std::endl;
                    local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                    center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
                    all_set[flat_index] = true;
                });
        }

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
        bool are_all_set = std::all_of(all_set.begin(), all_set.end(), [](bool i) { return i; });
        std::cout << "all_set are all set?: " << std::boolalpha << are_all_set << std::endl;
        // if (!are_all_set) {
        //     for (size_t i0 = 0; i0 < PADDED_STRIDE; i0++) {
        //         std::cout << "----- layer i0: " << i0 << " -----" << std::endl;
        //         for (size_t i1 = 0; i1 < PADDED_STRIDE; i1++) {
        //             for (size_t i2 = 0; i2 < PADDED_STRIDE; i2++) {
        //                 if (i2 > 0) {
        //                     std::cout << ", ";
        //                 }
        //                 const size_t flat_index =
        //                     i0 * PADDED_STRIDE * PADDED_STRIDE + i1 * PADDED_STRIDE + i2;
        //                 std::cout << all_set[flat_index];
        //             }
        //             std::cout << std::endl;
        //         }
        //         std::cout << "----- layer done -----" << std::endl;
        //     }
        // }
        std::cout << "result variables initialized" << std::endl;
    }

    m2m_interactions::m2m_interactions()
      : verbose(true) {
        // allocate input variables with padding (so that the interactions spheres are always valid
        // indices)

        if (verbose) {
            std::cout << "EXPANSION_COUNT_NOT_PADDED: " << EXPANSION_COUNT_NOT_PADDED << std::endl;
            std::cout << "EXPANSION_COUNT_PADDED: " << EXPANSION_COUNT_PADDED << std::endl;
        }
        local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
        center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

        iterate_inner_cells_padded([this](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            std::cout << "i is " << i << " i_unpadded is " << i_unpadded << std::endl;
            std::cout << "flat_index: " << flat_index
                      << " flat_index_unpadded: " << flat_index_unpadded << std::endl;
        });

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
    }

    void m2m_interactions::get_converted_local_expansions(std::vector<multipole>& M_ptr) {
        iterate_inner_cells_padded([this, &M_ptr](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            M_ptr[flat_index_unpadded] = local_expansions[flat_index];
        });
    }

    void m2m_interactions::get_converted_center_of_masses(
        std::vector<std::shared_ptr<std::vector<space_vector>>> com_ptr) {
        std::vector<space_vector>& com0 = *(com_ptr[0]);
        iterate_inner_cells_padded([this, &com0](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            // multiindex i_unpadded(i.x - INNER_CELLS_PADDING_DEPTH, i.y -
            // INNER_CELLS_PADDING_DEPTH,
            //     i.z - INNER_CELLS_PADDING_DEPTH);
            // size_t flat_index_unpadded = to_inner_flat_index_not_padded(i_unpadded);
            com0[flat_index_unpadded] = center_of_masses[flat_index];
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
