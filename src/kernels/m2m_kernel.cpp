#include "m2m_kernel.hpp"

#include "defs.hpp"
#include "interaction_types.hpp"
#include "options.hpp"
#include "struct_of_array_taylor.hpp"

#include <array>
#include <functional>

// std::vector<interaction_type> ilist_debugging;

extern options opts;

namespace octotiger {
namespace fmm {

    m2m_kernel::m2m_kernel(struct_of_array_data<expansion, real, 20>& local_expansions_SoA,
        struct_of_array_data<space_vector, real, 3>& center_of_masses_SoA,
        struct_of_array_data<expansion, real, 20>& potential_expansions_SoA,
        struct_of_array_data<space_vector, real, 3>& angular_corrections_SoA,
        gsolve_type type)
      :    // local_expansions(local_expansions),
      local_expansions_SoA(local_expansions_SoA)
      // , center_of_masses(center_of_masses)
      , center_of_masses_SoA(center_of_masses_SoA)
      // , potential_expansions(potential_expansions)
      , potential_expansions_SoA(potential_expansions_SoA)
      // , angular_corrections(angular_corrections)
      , angular_corrections_SoA(angular_corrections_SoA)
      , type(type)
      , theta_rec_sqared(sqr(1.0 / opts.theta)) {
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

                        // indices on coarser level (for outer stencil boundary)
                        // implicitly broadcasts to vector
                        multiindex<int_simd_vector> cell_index_coarse(cell_index);
                        for (size_t j = 0; j < int_simd_vector::Size; j++) {
                            cell_index_coarse.z[j] += j;
                        }
                        cell_index_coarse.transform_coarse();

                        // for (size_t inner_stencil_index = 0;
                        //      inner_stencil_index < STENCIL_BLOCKING &&
                        //      outer_stencil_index + inner_stencil_index < stencil.size();
                        //      inner_stencil_index += 1) {
                        this->single_interaction(cell_index, cell_flat_index, cell_index_coarse,
                            cell_index_unpadded, cell_flat_index_unpadded, stencil,
                            outer_stencil_index);
                        // }

                        // this->single_interaction(
                        //     cell_index, cell_flat_index, cell_index_coarse, cell_index_unpadded,
                        //     cell_flat_index_unpadded, stencil, outer_stencil_index
                        //     // interaction_partner_index, interaction_flat_partner_index
                        //     );
                    }
                }
            }
        }
    }

    // void m2m_kernel::apply_stencil_element(multiindex& stencil_element) {
    //     // execute kernel for individual interactions for every inner cell
    //     // use this object as a functor for the iteration
    //     iterate_inner_cells_padded_stencil(stencil_element, *this);

    //     // std::cout << "potential_expansions.size(): " <<
    //     potential_expansions.size() <<
    //     std::endl;
    //     // std::cout << "potential_expansions[0]: " << potential_expansions[0];
    //     // std::cout << std::endl;
    // }

}    // namespace fmm
}    // namespace octotiger
