#pragma once

#include "interaction_constants.hpp"
#include "interactions_iterators.hpp"
#include "multiindex.hpp"
#include "taylor.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

    class m2m_kernel
    {
    private:
        std::vector<expansion>& local_expansions;

        // com0 = *(com_ptr[0])
        std::vector<space_vector>& center_of_masses;

        // multipole expansion on this cell (L)
        std::vector<expansion>& potential_expansions;
        // angular momentum correction on this cell (L_c)
        std::vector<space_vector>& angular_corrections;

        gsolve_type type;

        //
        template <typename F>
        void iterate_inner_cells_padded_stencil(multiindex& stencil_element, F f) {
            for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                    for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        const multiindex cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                            i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                        const size_t cell_flat_index =
                            octotiger::fmm::to_inner_flat_index_padded(cell_index);
                        const multiindex cell_index_unpadded(i0, i1, i2);
                        const size_t cell_flat_index_unpadded =
                            to_inner_flat_index_not_padded(cell_index_unpadded);
                        const multiindex interaction_partner_index(cell_index.x + stencil_element.x,
                            cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);
                        const size_t interaction_flat_partner_index =
                            to_inner_flat_index_padded(interaction_partner_index);
                        f(cell_index, cell_flat_index, cell_index_unpadded,
                            cell_flat_index_unpadded, interaction_partner_index,
                            interaction_flat_partner_index);
                    }
                }
            }
        }

    public:
        m2m_kernel(std::vector<expansion>& local_expansions,
            std::vector<space_vector>& center_of_masses,
            std::vector<expansion>& potential_expansions,
            std::vector<space_vector>& angular_corrections, gsolve_type type);

        void apply_stencil_element(multiindex& m);
    };

}    // namespace fmm
}    // namespace octotiger
