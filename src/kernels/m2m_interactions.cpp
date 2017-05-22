#include "m2m_interactions.hpp"

#include "calculate_stencil.hpp"
#include "interactions_iterators.hpp"
#include "m2m_kernel.hpp"

#include <algorithm>

// Big picture questions:
// - use any kind of tiling?

namespace octotiger {
namespace fmm {

    m2m_interactions::m2m_interactions(std::vector<multipole>& M_ptr,
        std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
        // grid& g,
        std::vector<neighbor_gravity_type>& neighbors, gsolve_type type)
      : verbose(true)
      , type(type) {
        stencils = calculate_stencil();
        // std::vector<multipole>& M_ptr = g.get_M();
        // std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr = g.get_com_ptr();
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

        // for (const geo::direction& dir : geo::direction::full_set()) {
        //     std::cout << "dir 0: " << dir[0] << " 1: " << dir[1] << " 2: " << dir[2] <<
        //     std::endl;
        // for (neighbor_gravity_type& neighbor : neighbors) {
        for (const geo::direction& dir : geo::direction::full_set()) {
            neighbor_gravity_type& neighbor = neighbors[dir];
            std::cout << "dir: " << dir << std::endl;
            std::cout << "is neighbor monopole: " << std::boolalpha << neighbor.is_monopole << std::endl;
            if (!neighbor.data.M) {
                throw "neighbor M empty";
            }
            if (!neighbor.data.x) {
                throw "neighbor x empty";
            }

            std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
            std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
            iterate_inner_cells_padding(
                neighbor.direction,
                [this, neighbor_M_ptr, neighbor_com0](const multiindex& i, const size_t flat_index,
                    const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                    // local_expansions.at(flat_index) = neighbor_M_ptr.at(flat_index_unpadded);
                    center_of_masses.at(flat_index) = neighbor_com0.at(flat_index_unpadded);
                });
        }

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                expansion& e = potential_expansions.at(flat_index_unpadded);
                for (size_t j = 0; j < e.size(); j++) {
                    e[j] = 0.0;
                }
            });
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                space_vector& s = angular_corrections.at(flat_index_unpadded);
                for (size_t j = 0; j < s.size(); j++) {
                    s[j] = 0.0;
                }
            });
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
        // std::vector<multiindex>& stencil = stencils[0];
        // for (multiindex& m : stencil) {
        //     std::cout << "next stencil: " << m << std::endl;
        //     kernel.apply_stencil_element(m);
        // }
        kernel.apply_stencils(stencils);
    }

    void m2m_interactions::get_converted_local_expansions(std::vector<multipole>& M_ptr) {
        iterate_inner_cells_padded([this, &M_ptr](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            M_ptr[flat_index_unpadded] = local_expansions[flat_index];
        });
    }

    std::vector<expansion>& m2m_interactions::get_local_expansions() {
        return local_expansions;
    }

    void m2m_interactions::get_converted_center_of_masses(
        std::vector<std::shared_ptr<std::vector<space_vector>>> com_ptr) {
        std::vector<space_vector>& com0 = *(com_ptr[0]);
        iterate_inner_cells_padded([this, &com0](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            com0[flat_index_unpadded] = center_of_masses[flat_index];
        });
    }

    std::vector<space_vector>& m2m_interactions::get_center_of_masses() {
        return center_of_masses;
    }

    void m2m_interactions::get_converted_potential_expansions(std::vector<expansion>& L) {
        iterate_inner_cells_not_padded([this, &L](multiindex& i, size_t flat_index) {
            L[flat_index] = potential_expansions[flat_index];
        });
    }

    std::vector<expansion>& m2m_interactions::get_potential_expansions() {
        return potential_expansions;
    }

    void m2m_interactions::get_converted_angular_corrections(std::vector<space_vector>& L_c) {
        iterate_inner_cells_not_padded([this, &L_c](multiindex& i, size_t flat_index) {
            L_c[flat_index] = angular_corrections[flat_index];
        });
    }

    std::vector<space_vector>& m2m_interactions::get_angular_corrections() {
        return angular_corrections;
    }

    void m2m_interactions::print_potential_expansions() {
        print_layered_not_padded(true, [this](const multiindex& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
        });
    }

    void m2m_interactions::print_angular_corrections() {
        print_layered_not_padded(true, [this](const multiindex& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->angular_corrections[flat_index];
        });
    }

    void m2m_interactions::print_local_expansions() {
        print_layered_inner_padded(
            true, [this](const multiindex& i, const size_t flat_index, const multiindex& i_unpadded,
                      const size_t flat_index_unpadded) {
                std::cout << " " << this->local_expansions[flat_index];
            });
    }

    void m2m_interactions::print_center_of_masses() {
        print_layered_padded(true, [this](const multiindex& i, const size_t flat_index) {
            std::cout << this->center_of_masses[flat_index];
        });
    }

}    // namespace fmm
}    // namespace octotiger
