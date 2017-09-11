#include "m2m_interactions.hpp"

#include "calculate_stencil.hpp"
#include "interactions_iterators.hpp"
#include "m2m_cuda.hpp"
#include "m2m_kernel.hpp"
//
// #include "cuda/cuda_helper.h"
//
#include <algorithm>

#include "options.hpp"

// Big picture questions:
// - use any kind of tiling?

size_t total_neighbors = 0;
size_t missing_neighbors = 0;

extern options opts;

// required for cuda
extern taylor<4, real> factor;
extern taylor<4, real> factor_half;
extern taylor<4, real> factor_sixth;

namespace octotiger {
namespace fmm {

    extern std::vector<octotiger::fmm::multiindex<>> stencil;

    m2m_interactions::m2m_interactions(std::vector<multipole>& M_ptr,
        std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
        // grid& g,
        std::vector<neighbor_gravity_type>& neighbors, gsolve_type type)
      : neighbor_empty(27)
      , type(type) {
        local_expansions = std::vector<expansion>(EXPANSION_COUNT_PADDED);
        center_of_masses = std::vector<space_vector>(EXPANSION_COUNT_PADDED);

        std::vector<space_vector> const& com0 = *(com_ptr[0]);

        iterate_inner_cells_padded(
            [this, M_ptr, com0](const multiindex<>& i, const size_t flat_index,
                const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                local_expansions.at(flat_index) = M_ptr.at(flat_index_unpadded);
                center_of_masses.at(flat_index) = com0.at(flat_index_unpadded);
            });

        total_neighbors += 27;

        size_t current_missing = 0;
        size_t current_monopole = 0;

        for (size_t i = 0; i < neighbor_empty.size(); i++) {
            neighbor_empty[i] = false;
        }

        for (const geo::direction& dir : geo::direction::full_set()) {
            // don't use neighbor.direction, is always zero for empty cells!
            neighbor_gravity_type& neighbor = neighbors[dir];

            // this dir is setup as a multipole
            if (!neighbor.is_monopole) {
                if (!neighbor.data.M) {
                    // TODO: ask Dominic why !is_monopole and stuff still empty
                    iterate_inner_cells_padding(
                        dir, [this](const multiindex<>& i, const size_t flat_index,
                                 const multiindex<>&, const size_t) {
                            // initializes whole expansion, relatively expansion
                            local_expansions.at(flat_index) = 0.0;
                            // initializes x,y,z vector
                            center_of_masses.at(flat_index) = 0.0;
                        });
                    missing_neighbors += 1;
                    current_missing += 1;
                    neighbor_empty[dir.flat_index_with_center()] = true;
                } else {
                    std::vector<multipole>& neighbor_M_ptr = *(neighbor.data.M);
                    std::vector<space_vector>& neighbor_com0 = *(neighbor.data.x);
                    iterate_inner_cells_padding(
                        dir, [this, neighbor_M_ptr, neighbor_com0](const multiindex<>& i,
                                 const size_t flat_index, const multiindex<>& i_unpadded,
                                 const size_t flat_index_unpadded) {
                            local_expansions.at(flat_index) =
                                neighbor_M_ptr.at(flat_index_unpadded);
                            center_of_masses.at(flat_index) = neighbor_com0.at(flat_index_unpadded);
                        });
                }
            } else {
                // in case of monopole, boundary becomes padding in that direction
                // TODO: setting everything to zero might not be correct to create zero potentials
                iterate_inner_cells_padding(
                    dir, [this](const multiindex<>& i, const size_t flat_index, const multiindex<>&,
                             const size_t) {
                        // initializes whole expansion, relatively expansion
                        local_expansions.at(flat_index) = 0.0;
                        // initializes x,y,z vector
                        center_of_masses.at(flat_index) = 0.0;
                    });
                missing_neighbors += 1;
                current_monopole += 1;
                neighbor_empty[dir.flat_index_with_center()] = true;
            }
        }

        neighbor_empty[13] = false;

        // std::cout << current_monopole << " monopoles, " << current_monopole << " missing
        // neighbors"
        //           << std::endl;

        // std::cout << std::boolalpha;
        // for (size_t i = 0; i < 27; i++) {
        //     if (i > 0) {
        //         std::cout << ", ";
        //     }
        //     std::cout << i << " -> " << neighbor_empty[i];
        // }
        // std::cout << std::endl;

        // allocate output variables without padding
        potential_expansions = std::vector<expansion>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                expansion& e = potential_expansions.at(flat_index_unpadded);
                e = 0.0;
            });
        angular_corrections = std::vector<space_vector>(EXPANSION_COUNT_NOT_PADDED);
        // TODO/BUG: expansion don't initialize to zero by default
        iterate_inner_cells_not_padded(
            [this](const multiindex<>& i_unpadded, const size_t flat_index_unpadded) {
                space_vector& s = angular_corrections.at(flat_index_unpadded);
                s = 0.0;
            });

        // std::cout << "local_expansions:" << std::endl;
        // this->print_local_expansions();
        // std::cout << "center_of_masses:" << std::endl;
        // this->print_center_of_masses();
    }

    void m2m_interactions::compute_interactions() {
        struct_of_array_data<real, 20, ENTRIES_PADDED, SOA_PADDING> local_expansions_SoA =
            struct_of_array_data<real, 20, ENTRIES_PADDED, SOA_PADDING>::from_vector(
                local_expansions);
        struct_of_array_data<real, 3, ENTRIES_PADDED, SOA_PADDING> center_of_masses_SoA =
            struct_of_array_data<real, 3, ENTRIES_PADDED, SOA_PADDING>::from_vector(
                center_of_masses);
        // struct_of_array_data<real, 20, ENTRIES_NOT_PADDED, SOA_PADDING> potential_expansions_SoA
        // =
        //     struct_of_array_data<real, 20, ENTRIES_NOT_PADDED, SOA_PADDING>::from_vector(
        //         potential_expansions);
        // struct_of_array_data<real, 3, ENTRIES_NOT_PADDED, SOA_PADDING> angular_corrections_SoA =
        //     struct_of_array_data<real, 3, ENTRIES_NOT_PADDED, SOA_PADDING>::from_vector(
        //         angular_corrections);

        struct_of_array_data<real, 20, ENTRIES_NOT_PADDED, SOA_PADDING> potential_expansions_SoA;
        for (size_t i = 0; i < potential_expansions_SoA.size(); i++) {
            potential_expansions_SoA.get_underlying_pointer()[i] = 0.0;
        }
        struct_of_array_data<real, 3, ENTRIES_NOT_PADDED, SOA_PADDING> angular_corrections_SoA;
        for (size_t i = 0; i < angular_corrections_SoA.size(); i++) {
            angular_corrections_SoA.get_underlying_pointer()[i] = 0.0;
        }

        struct_of_array_data<real, 20, ENTRIES_NOT_PADDED, SOA_PADDING>
            potential_expansions_SoA_cuda;
        for (size_t i = 0; i < potential_expansions_SoA.size(); i++) {
            potential_expansions_SoA_cuda.get_underlying_pointer()[i] = 0.0;
        }
        struct_of_array_data<real, 3, ENTRIES_NOT_PADDED, SOA_PADDING> angular_corrections_SoA_cuda;
        for (size_t i = 0; i < angular_corrections_SoA.size(); i++) {
            angular_corrections_SoA_cuda.get_underlying_pointer()[i] = 0.0;
        }

        m2m_kernel kernel(
            // local_expansions_SoA,
            //               center_of_masses_SoA, potential_expansions_SoA,
            // angular_corrections_SoA,
            neighbor_empty, type);

        auto start = std::chrono::high_resolution_clock::now();

        // std::cout << "local_expansions_SoA on HOST:" << std::endl;
        // for (size_t i = 0; i < local_expansions_SoA.size(); i++) {
        //     if (i > 0) {
        //         std::cout << ", ";
        //     }
        //     std::cout << local_expansions_SoA.get_underlying_pointer()[i];
        // }
        // std::cout << std::endl;

        cuda::m2m_cuda cuda_kernel;
        cuda_kernel.compute_interactions(local_expansions_SoA, center_of_masses_SoA,
            potential_expansions_SoA_cuda, angular_corrections_SoA_cuda, opts.theta,
            factor.get_array(), factor_half.get_array(), factor_sixth.get_array(), type);

        // auto interaction_future = cuda_helper.get_future();
        // interaction_future.get();
        // std::cout << "The cuda future has completed successfully" << std::endl;

        kernel.apply_stencil(local_expansions_SoA, center_of_masses_SoA, potential_expansions_SoA,
            angular_corrections_SoA, stencil);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "potential_expansions_SoA:" << std::endl;
        std::cout << "first potential_expansions_SoA_cuda:" << std::endl;
        std::cout << "0: " << potential_expansions_SoA_cuda.value<0>(0) << std::endl;
        std::cout << "1: " << potential_expansions_SoA_cuda.value<1>(0) << std::endl;
        std::cout << "2: " << potential_expansions_SoA_cuda.value<2>(0) << std::endl;
        std::cout << "3: " << potential_expansions_SoA_cuda.value<3>(0) << std::endl;
        std::cout << "4: " << potential_expansions_SoA_cuda.value<4>(0) << std::endl;
        std::cout << "5: " << potential_expansions_SoA_cuda.value<5>(0) << std::endl;
        std::cout << "6: " << potential_expansions_SoA_cuda.value<6>(0) << std::endl;
        std::cout << "7: " << potential_expansions_SoA_cuda.value<7>(0) << std::endl;
        std::cout << "8: " << potential_expansions_SoA_cuda.value<8>(0) << std::endl;
        std::cout << "9: " << potential_expansions_SoA_cuda.value<9>(0) << std::endl;
        std::cout << "10: " << potential_expansions_SoA_cuda.value<10>(0) << std::endl;
        std::cout << "11: " << potential_expansions_SoA_cuda.value<11>(0) << std::endl;
        std::cout << "12: " << potential_expansions_SoA_cuda.value<12>(0) << std::endl;
        std::cout << "13: " << potential_expansions_SoA_cuda.value<13>(0) << std::endl;
        std::cout << "14: " << potential_expansions_SoA_cuda.value<14>(0) << std::endl;
        std::cout << "15: " << potential_expansions_SoA_cuda.value<15>(0) << std::endl;
        std::cout << "16: " << potential_expansions_SoA_cuda.value<16>(0) << std::endl;
        std::cout << "17: " << potential_expansions_SoA_cuda.value<17>(0) << std::endl;
        std::cout << "18: " << potential_expansions_SoA_cuda.value<18>(0) << std::endl;
        std::cout << "19: " << potential_expansions_SoA_cuda.value<19>(0) << std::endl;

        std::cout << "first potential_expansions_SoA:" << std::endl;
        std::cout << "0: " << potential_expansions_SoA.value<0>(0) << std::endl;
        std::cout << "1: " << potential_expansions_SoA.value<1>(0) << std::endl;
        std::cout << "2: " << potential_expansions_SoA.value<2>(0) << std::endl;
        std::cout << "3: " << potential_expansions_SoA.value<3>(0) << std::endl;
        std::cout << "4: " << potential_expansions_SoA.value<4>(0) << std::endl;
        std::cout << "5: " << potential_expansions_SoA.value<5>(0) << std::endl;
        std::cout << "6: " << potential_expansions_SoA.value<6>(0) << std::endl;
        std::cout << "7: " << potential_expansions_SoA.value<7>(0) << std::endl;
        std::cout << "8: " << potential_expansions_SoA.value<8>(0) << std::endl;
        std::cout << "9: " << potential_expansions_SoA.value<9>(0) << std::endl;
        std::cout << "10: " << potential_expansions_SoA.value<10>(0) << std::endl;
        std::cout << "11: " << potential_expansions_SoA.value<11>(0) << std::endl;
        std::cout << "12: " << potential_expansions_SoA.value<12>(0) << std::endl;
        std::cout << "13: " << potential_expansions_SoA.value<13>(0) << std::endl;
        std::cout << "14: " << potential_expansions_SoA.value<14>(0) << std::endl;
        std::cout << "15: " << potential_expansions_SoA.value<15>(0) << std::endl;
        std::cout << "16: " << potential_expansions_SoA.value<16>(0) << std::endl;
        std::cout << "17: " << potential_expansions_SoA.value<17>(0) << std::endl;
        std::cout << "18: " << potential_expansions_SoA.value<18>(0) << std::endl;
        std::cout << "19: " << potential_expansions_SoA.value<19>(0) << std::endl;
        for (size_t i = 0; i < potential_expansions_SoA.size(); i++) {
            if (potential_expansions_SoA.get_underlying_pointer()[i] -
                    potential_expansions_SoA_cuda.get_underlying_pointer()[i] >
                1E-6) {
                std::cout << "ref: " << potential_expansions_SoA.get_underlying_pointer()[i]
                          << " mine: " << potential_expansions_SoA_cuda.get_underlying_pointer()[i]
                          << " abs err: "
                          << (potential_expansions_SoA.get_underlying_pointer()[i] -
                                 potential_expansions_SoA_cuda.get_underlying_pointer()[i])
                          << " flat_index: " << i << std::endl;
                std::cout << std::flush;
                throw;
            }
        }

        std::cout << "angular_corrections_SoA:" << std::endl;
        for (size_t i = 0; i < angular_corrections_SoA.size(); i++) {
            if (angular_corrections_SoA.get_underlying_pointer()[i] -
                    angular_corrections_SoA_cuda.get_underlying_pointer()[i] >
                1E-6) {
                std::cout << "ref: " << angular_corrections_SoA.get_underlying_pointer()[i]
                          << " mine: " << angular_corrections_SoA_cuda.get_underlying_pointer()[i]
                          << " abs err: "
                          << (angular_corrections_SoA.get_underlying_pointer()[i] -
                                 angular_corrections_SoA_cuda.get_underlying_pointer()[i])
                          << " flat_index: " << i << std::endl;
                std::cout << std::flush;
                throw;
            }
        }

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "new interaction kernel (apply only, ms): " << duration.count() << std::endl;

        // throw;

        // TODO: remove this after finalizing conversion
        // copy back SoA data into non-SoA result
        potential_expansions_SoA_cuda.to_non_SoA(potential_expansions);
        angular_corrections_SoA_cuda.to_non_SoA(angular_corrections);
    }

    std::vector<expansion>& m2m_interactions::get_local_expansions() {
        return local_expansions;
    }

    std::vector<space_vector>& m2m_interactions::get_center_of_masses() {
        return center_of_masses;
    }

    std::vector<expansion>& m2m_interactions::get_potential_expansions() {
        return potential_expansions;
    }

    std::vector<space_vector>& m2m_interactions::get_angular_corrections() {
        return angular_corrections;
    }

    void m2m_interactions::print_potential_expansions() {
        print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->potential_expansions[flat_index][0];
        });
    }

    void m2m_interactions::print_angular_corrections() {
        print_layered_not_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << " (" << i << ") =[0] " << this->angular_corrections[flat_index];
        });
    }

    void m2m_interactions::print_local_expansions() {
        print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << " " << this->local_expansions[flat_index];
        });
    }

    void m2m_interactions::print_center_of_masses() {
        print_layered_padded(true, [this](const multiindex<>& i, const size_t flat_index) {
            std::cout << this->center_of_masses[flat_index];
        });
    }

    void m2m_interactions::add_to_potential_expansions(std::vector<expansion>& L) {
        iterate_inner_cells_not_padded([this, &L](multiindex<>& i, size_t flat_index) {
            potential_expansions[flat_index] += L[flat_index];
        });
    }

    void m2m_interactions::add_to_center_of_masses(std::vector<space_vector>& L_c) {
        iterate_inner_cells_not_padded([this, &L_c](multiindex<>& i, size_t flat_index) {
            center_of_masses[flat_index] += L_c[flat_index];
        });
    }

}    // namespace fmm
}    // namespace octotiger
