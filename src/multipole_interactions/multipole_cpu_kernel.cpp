#include <array>
#include <functional>

#include "../common_kernel/helper.hpp"
#include "../common_kernel/interaction_constants.hpp"
#include "../common_kernel/struct_of_array_data.hpp"
#include "compute_kernel_templates.hpp"
#include "defs.hpp"
#include "interaction_types.hpp"
#include "options.hpp"

#include "multipole_cpu_kernel.hpp"

extern options opts;

namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        multipole_cpu_kernel::multipole_cpu_kernel(void)
          : theta_rec_squared(sqr(1.0 / opts.theta)) {
            for (size_t i = 0; i < m2m_int_vector::size(); i++) {
                offset_vector[i] = i;
            }
        }

        void multipole_cpu_kernel::apply_stencil(const struct_of_array_data<expansion, real, 20,
                                                     ENTRIES, SOA_PADDING>& local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<real>& mons, const two_phase_stencil& stencil, gsolve_type type) {
            for (size_t outer_stencil_index = 0;
                 outer_stencil_index < stencil.stencil_elements.size();
                 outer_stencil_index += STENCIL_BLOCKING) {
                for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
                    for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                        // for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                        for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION;
                             i2 += m2m_vector::size()) {
                            const multiindex<> cell_index(i0 + INNER_CELLS_PADDING_DEPTH,
                                i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                            // BUG: indexing has to be done with uint32_t because of Vc limitation
                            const int64_t cell_flat_index =
                                to_flat_index_padded(cell_index);    // iii0...
                            const multiindex<> cell_index_unpadded(i0, i1, i2);
                            const int64_t cell_flat_index_unpadded =
                                to_inner_flat_index_not_padded(cell_index_unpadded);

                            // indices on coarser level (for outer stencil boundary)
                            // implicitly broadcasts to vector
                            multiindex<m2m_int_vector> cell_index_coarse(cell_index);
                            for (size_t j = 0; j < m2m_int_vector::size(); j++) {
                                cell_index_coarse.z[j] += j;
                            }
                            // note that this is the same for groups of 2x2x2 elements
                            // -> maps to the same for some SIMD lanes
                            cell_index_coarse.transform_coarse();

                            if (type == RHO) {
                                this->blocked_interaction_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index);
                                this->blocked_angular_corrections(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index);
                            } else {
                                this->blocked_interaction_non_rho(local_expansions_SoA,
                                    center_of_masses_SoA, potential_expansions_SoA,
                                    angular_corrections_SoA, mons, cell_index, cell_flat_index,
                                    cell_index_coarse, cell_index_unpadded,
                                    cell_flat_index_unpadded, stencil, outer_stencil_index);
                            }
                        }
                    }
                }
            }
        }

        void multipole_cpu_kernel::blocked_interaction_rho(
            const struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS,
                SOA_PADDING>& __restrict__ angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded, const two_phase_stencil& __restrict__ stencil,
            const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];

            m2m_vector Y[3];

            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.stencil_elements.size();
                 inner_stencil_index += 1) {
                const bool phase_one =
                    stencil.stencil_phase_indicator[outer_stencil_index + inner_stencil_index];
                const multiindex<>& stencil_element =
                    stencil.stencil_elements[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                m2m_vector m_partner[20];
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                m2m_vector::mask_type mask_phase_one(phase_one);

                update_mask<0, m2m_vector::mask_type, std::vector<real>, m2m_vector(&)[20]>(
                    mask, mask_phase_one, mons, m_partner, interaction_partner_flat_index);

                unrolled_SoA_load<0, 19,
                    struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>,
                    m2m_vector(&)[20], m2m_vector::mask_type>(
                    local_expansions_SoA, m_partner, mask, interaction_partner_flat_index);

                compute_kernel_basic_non_rho(X, Y, m_partner, tmpstore,
                    [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                        return Vc::max(one, two);
                    });
            }
            if (changed_data) {
                unrolled_SoA_write<0, 19, m2m_vector(&)[20],
                    struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>>(
                    tmpstore, potential_expansions_SoA, cell_flat_index_unpadded);
            }
        }

        void multipole_cpu_kernel::blocked_interaction_non_rho(
            const struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>&
                local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES, SOA_PADDING>&
                center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>&
                potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS, SOA_PADDING>&
                angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& cell_index,
            const size_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
            const multiindex<>& cell_index_unpadded, const size_t cell_flat_index_unpadded,
            const two_phase_stencil& stencil, const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];

            m2m_vector Y[3];

            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.stencil_elements.size();
                 inner_stencil_index += 1) {
                const bool phase_one =
                    stencil.stencil_phase_indicator[outer_stencil_index + inner_stencil_index];
                const multiindex<>& stencil_element =
                    stencil.stencil_elements[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                m2m_vector m_partner[20];
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                m2m_vector::mask_type mask_phase_one(phase_one);

                update_mask<0, m2m_vector::mask_type, std::vector<real>, m2m_vector(&)[20]>(
                    mask, mask_phase_one, mons, m_partner, interaction_partner_flat_index);

                unrolled_SoA_load<0, 19,
                    struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>,
                    m2m_vector(&)[20], m2m_vector::mask_type>(
                    local_expansions_SoA, m_partner, mask, interaction_partner_flat_index);

                compute_kernel_non_rho(X, Y, m_partner, tmpstore,
                    [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                        return Vc::max(one, two);
                    });
            }
            if (changed_data) {
                unrolled_SoA_write<0, 19, m2m_vector(&)[20],
                    struct_of_array_data<expansion, real, 20, INNER_CELLS, SOA_PADDING>>(
                    tmpstore, potential_expansions_SoA, cell_flat_index_unpadded);
            }
        }

        void multipole_cpu_kernel::blocked_angular_corrections(
            const struct_of_array_data<expansion, real, 20, ENTRIES,
                SOA_PADDING>& __restrict__ local_expansions_SoA,
            const struct_of_array_data<space_vector, real, 3, ENTRIES,
                SOA_PADDING>& __restrict__ center_of_masses_SoA,
            struct_of_array_data<expansion, real, 20, INNER_CELLS,
                SOA_PADDING>& __restrict__ potential_expansions_SoA,
            struct_of_array_data<space_vector, real, 3, INNER_CELLS,
                SOA_PADDING>& __restrict__ angular_corrections_SoA,
            const std::vector<real>& mons, const multiindex<>& __restrict__ cell_index,
            const size_t cell_flat_index,
            const multiindex<m2m_int_vector>& __restrict__ cell_index_coarse,
            const multiindex<>& __restrict__ cell_index_unpadded,
            const size_t cell_flat_index_unpadded, const two_phase_stencil& __restrict__ stencil,
            const size_t outer_stencil_index) {
            m2m_vector X[3];
            X[0] = center_of_masses_SoA.value<0>(cell_flat_index);
            X[1] = center_of_masses_SoA.value<1>(cell_flat_index);
            X[2] = center_of_masses_SoA.value<2>(cell_flat_index);
            m2m_vector tmpstore[20];
            m2m_vector tmp_corrections[3];

            m2m_vector m_cell[11];
            m_cell[0] = local_expansions_SoA.value<0>(cell_flat_index);
            m_cell[1] = local_expansions_SoA.value<10>(cell_flat_index);
            m_cell[2] = local_expansions_SoA.value<11>(cell_flat_index);
            m_cell[3] = local_expansions_SoA.value<12>(cell_flat_index);
            m_cell[4] = local_expansions_SoA.value<13>(cell_flat_index);
            m_cell[5] = local_expansions_SoA.value<14>(cell_flat_index);
            m_cell[6] = local_expansions_SoA.value<15>(cell_flat_index);
            m_cell[7] = local_expansions_SoA.value<16>(cell_flat_index);
            m_cell[8] = local_expansions_SoA.value<17>(cell_flat_index);
            m_cell[9] = local_expansions_SoA.value<18>(cell_flat_index);
            m_cell[10] = local_expansions_SoA.value<19>(cell_flat_index);

            m2m_vector Y[3];

            bool changed_data = false;
            for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
                 outer_stencil_index + inner_stencil_index < stencil.stencil_elements.size();
                 inner_stencil_index += 1) {
                const bool phase_one =
                    stencil.stencil_phase_indicator[outer_stencil_index + inner_stencil_index];
                const multiindex<>& stencil_element =
                    stencil.stencil_elements[outer_stencil_index + inner_stencil_index];
                const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                    cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

                const size_t interaction_partner_flat_index =
                    to_flat_index_padded(interaction_partner_index);    // iii1n

                // implicitly broadcasts to vector
                multiindex<m2m_int_vector> interaction_partner_index_coarse(
                    interaction_partner_index);
                interaction_partner_index_coarse.z += offset_vector;
                // note that this is the same for groups of 2x2x2 elements
                // -> maps to the same for some SIMD lanes
                interaction_partner_index_coarse.transform_coarse();

                m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                    cell_index_coarse, interaction_partner_index_coarse);

                m2m_vector theta_c_rec_squared =
                    // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                    Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);

                m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

                if (Vc::none_of(mask)) {
                    continue;
                }
                changed_data = true;

                m2m_vector m_partner[20];
                Y[0] = center_of_masses_SoA.value<0>(interaction_partner_flat_index);
                Y[1] = center_of_masses_SoA.value<1>(interaction_partner_flat_index);
                Y[2] = center_of_masses_SoA.value<2>(interaction_partner_flat_index);

                m2m_vector::mask_type mask_phase_one(phase_one);

                update_mask<0, m2m_vector::mask_type, std::vector<real>, m2m_vector(&)[20]>(
                    mask, mask_phase_one, mons, m_partner, interaction_partner_flat_index);

                unrolled_SoA_load<0, 19,
                    struct_of_array_data<expansion, real, 20, ENTRIES, SOA_PADDING>,
                    m2m_vector(&)[20], m2m_vector::mask_type>(
                    local_expansions_SoA, m_partner, mask, interaction_partner_flat_index);

                compute_angular_corrections(X, Y, m_partner, tmp_corrections, m_cell,
                    [](const m2m_vector& one, const m2m_vector& two) -> m2m_vector {
                        return Vc::max(one, two);
                    });
            }
            if (changed_data) {
                tmp_corrections[0] =
                    tmp_corrections[0] + angular_corrections_SoA.value<0>(cell_flat_index_unpadded);
                tmp_corrections[1] =
                    tmp_corrections[1] + angular_corrections_SoA.value<1>(cell_flat_index_unpadded);
                tmp_corrections[2] =
                    tmp_corrections[2] + angular_corrections_SoA.value<2>(cell_flat_index_unpadded);
                tmp_corrections[0].memstore(
                    angular_corrections_SoA.pointer<0>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmp_corrections[1].memstore(
                    angular_corrections_SoA.pointer<1>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
                tmp_corrections[2].memstore(
                    angular_corrections_SoA.pointer<2>(cell_flat_index_unpadded),
                    Vc::flags::element_aligned);
            }
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
