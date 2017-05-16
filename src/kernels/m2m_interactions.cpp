#include "m2m_interactions.hpp"

#include "calculate_stencil.hpp"
#include "interactions_iterators.hpp"
#include "m2m_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {

    m2m_interactions::m2m_interactions(
        grid& g, std::vector<node_server::neighbor_gravity_type>& neighbors, gsolve_type type)
      : verbose(true)
      , type(type) {
        stencil = calculate_stencil();
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

        std::vector<space_vector> const& com0 = *(com_ptr[0]);

        iterate_inner_cells_padded([this, M_ptr, com0](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            // std::cout << "flat_index: " << flat_index
            //           << " flat_index_unpadded: " << flat_index_unpadded << std::endl;
            local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
            center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
        });

        for (const geo::direction& dir : geo::direction::full_set()) {
            std::cout << "dir 0: " << dir[0] << " 1: " << dir[1] << " 2: " << dir[2] << std::endl;
            iterate_inner_cells_padding(
                dir, [this, M_ptr, com0](const multiindex& i, const size_t flat_index,
                         const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                    local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                    center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
                });
        }

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
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

    void m2m_interactions::compute_interactions() {
        m2m_kernel kernel(
            local_expansions, center_of_masses, potential_expansions, angular_corrections, type);
        for (multiindex& m : stencil) {
            kernel.apply_stencil_element(m);
        }
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

    void m2m_interactions::print_potential_expansions() {
        iterate_inner_cells_not_padded([this](multiindex& i, size_t flat_index) {
            if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
            }
            // std::cout << this->potential_expansions[flat_index];
	    if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
		std::cout << ", ";
	    }
	    std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
            if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                std::cout << std::endl;
            }
        });
    }

    void m2m_interactions::print_angular_corrections() {
        iterate_inner_cells_not_padded([this](multiindex& i, size_t flat_index) {
            if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
            }
            // std::cout << this->potential_expansions[flat_index];
	    if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
		std::cout << ", ";
	    }
	    std::cout << " (" << i << ") = " << this->angular_corrections[flat_index];
            if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                std::cout << std::endl;
            }
        });
	}

}    // namespace fmm
}    // namespace octotiger
