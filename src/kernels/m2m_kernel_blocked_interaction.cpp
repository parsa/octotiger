#include "m2m_kernel.hpp"

#include "grid_flattened_indices.hpp"
#include "helper.hpp"
#include "m2m_taylor_set_basis.hpp"
#include "struct_of_array_taylor.hpp"

extern taylor<4, real> factor;

taylor<4, m2m_vector> factor_half_v;
taylor<4, m2m_vector> factor_sixth_v;

namespace octotiger {
namespace fmm {

    // TODO:
    // - check codegen and fix in Vc
    // - check for amount of temporaries
    // - try to replace expensive operations like sqrt
    // - remove all sqr()
    // - increase INX

    void m2m_kernel::blocked_interaction_rho(const multiindex<>& cell_index,
        const int64_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
        const multiindex<>& cell_index_unpadded, const int64_t cell_flat_index_unpadded,
        const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index) {
        // TODO: should change name to something better (not taylor, but space_vector)
        // struct_of_array_taylor<space_vector, real, 3> X =
        //     center_of_masses_SoA.get_view(cell_flat_index);

        struct_of_array_iterator<space_vector, real, 3> X(center_of_masses_SoA, cell_flat_index);
        for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
             outer_stencil_index + inner_stencil_index < stencil.size();
             inner_stencil_index += 1) {
            const multiindex<>& stencil_element =
                stencil[outer_stencil_index + inner_stencil_index];
            const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

            const int64_t interaction_partner_flat_index =
                to_flat_index_padded(interaction_partner_index);

            // check whether all vector elements are in empty border
            if (vector_is_empty[interaction_partner_flat_index]) {
                continue;
            }

            // implicitly broadcasts to vector
            multiindex<m2m_int_vector> interaction_partner_index_coarse(interaction_partner_index);
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

            struct_of_array_iterator<space_vector, real, 3> Y(
                center_of_masses_SoA, interaction_partner_flat_index);

            // distance between cells in all dimensions
            // TODO: replace by m2m_vector for vectorization or get rid of temporary
            m2m_vector dX[NDIM];
            dX[0] = X.value() - Y.value();
            X++;
            Y++;
            dX[1] = X.value() - Y.value();
            X++;
            Y++;
            dX[2] = X.value() - Y.value();

            // reset X for next iteration
            X.decrement(2);

            expansion_v m_partner;

            {
                struct_of_array_iterator<expansion, real, 20> m_partner_iterator(
                    local_expansions_SoA, interaction_partner_flat_index);
                m_partner[0] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[1] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[2] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[3] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[4] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[5] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[6] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[7] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[8] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[9] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[10] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[11] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[12] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[13] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[14] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[15] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[16] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[17] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[18] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[19] = m_partner_iterator.value();
                m_partner_iterator++;
            }

            // R_i in paper is the dX in the code
            // D is taylor expansion value for a given X expansion of the gravitational
            // potential
            // (multipole expansion)

            // calculates all D-values, calculate all coefficients of 1/r (not the
            // potential),
            // formula (6)-(9) and (19)
            D_split D_calculator(dX);
            // TODO: this translates to an initialization loop, bug!
            m2m_vector D_lower[20];
            D_calculator.calculate_D_lower(D_lower);

            // expansion_v current_potential;
            // TODO: this translates to an initialization loop, bug!
            m2m_vector cur_pot[10];

            // 10-19 are not cached!

            // the following loops calculate formula (10), potential from B->A

            cur_pot[0] = m_partner[0] * D_lower[0];
            cur_pot[1] = m_partner[0] * D_lower[1];
            cur_pot[2] = m_partner[0] * D_lower[2];
            cur_pot[3] = m_partner[0] * D_lower[3];

            cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half_v[4]);
            cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half_v[4]);
            cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half_v[4]);
            cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half_v[4]);

            cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half_v[5]);
            cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half_v[5]);
            cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half_v[5]);
            cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half_v[5]);

            cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half_v[6]);
            cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half_v[6]);
            cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half_v[6]);
            cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half_v[6]);

            cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half_v[7]);
            cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half_v[7]);
            cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half_v[7]);
            cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half_v[7]);

            cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half_v[8]);
            cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half_v[8]);
            cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half_v[8]);
            cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half_v[8]);

            cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half_v[9]);
            cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half_v[9]);
            cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half_v[9]);
            cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half_v[9]);

            cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth_v[10]);
            cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth_v[11]);
            cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth_v[12]);
            cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth_v[13]);
            cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth_v[14]);
            cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth_v[15]);
            cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth_v[16]);
            cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth_v[17]);
            cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth_v[18]);
            cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth_v[19]);

            cur_pot[4] = m_partner[0] * D_lower[4];
            cur_pot[5] = m_partner[0] * D_lower[5];
            cur_pot[6] = m_partner[0] * D_lower[6];
            cur_pot[7] = m_partner[0] * D_lower[7];
            cur_pot[8] = m_partner[0] * D_lower[8];
            cur_pot[9] = m_partner[0] * D_lower[9];

            cur_pot[4] -= m_partner[1] * D_lower[10];
            cur_pot[5] -= m_partner[1] * D_lower[11];
            cur_pot[6] -= m_partner[1] * D_lower[12];
            cur_pot[7] -= m_partner[1] * D_lower[13];
            cur_pot[8] -= m_partner[1] * D_lower[14];
            cur_pot[9] -= m_partner[1] * D_lower[15];

            cur_pot[4] -= m_partner[2] * D_lower[11];
            cur_pot[5] -= m_partner[2] * D_lower[13];
            cur_pot[6] -= m_partner[2] * D_lower[14];
            cur_pot[7] -= m_partner[2] * D_lower[16];
            cur_pot[8] -= m_partner[2] * D_lower[17];
            cur_pot[9] -= m_partner[2] * D_lower[18];

            cur_pot[4] -= m_partner[3] * D_lower[12];
            cur_pot[5] -= m_partner[3] * D_lower[14];
            cur_pot[6] -= m_partner[3] * D_lower[15];
            cur_pot[7] -= m_partner[3] * D_lower[17];
            cur_pot[8] -= m_partner[3] * D_lower[18];
            cur_pot[9] -= m_partner[3] * D_lower[19];

            struct_of_array_iterator<expansion, real, 20> current_potential_result(
                potential_expansions_SoA, cell_flat_index_unpadded);

            m2m_vector tmp = current_potential_result.value() + cur_pot[0];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;

            tmp = current_potential_result.value() + cur_pot[1];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[2];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[3];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;

            tmp = current_potential_result.value() + cur_pot[4];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[5];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[6];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[7];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[8];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[9];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;    // don't remove this

            tmp = current_potential_result.value() + m_partner[0] * D_lower[10];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[11];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[12];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[13];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[14];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[15];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[16];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[17];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[18];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[19];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);

            ////////////// angular momentum correction, if enabled /////////////////////////

            // Initalize moments and momentum
            // this branch computes the angular momentum correction, (20) in the
            // paper divide by mass of other cell
            struct_of_array_iterator<expansion, real, 20> m_cell_iterator(
                local_expansions_SoA, cell_flat_index);

            m2m_vector const n0_constant = m_partner[0] / m_cell_iterator.value();

            m_cell_iterator.increment(10);

            // calculating the coefficients for formula (M are the octopole moments)
            // the coefficients are calculated in (17) and (18)
            struct_of_array_iterator<space_vector, real, 3> current_angular_correction_result(
                angular_corrections_SoA, cell_flat_index_unpadded);

            m2m_vector D_upper[15];
            // D_calculator.calculate_D_upper(D_upper);

            m2m_vector current_angular_correction[NDIM];
            current_angular_correction[0] = 0.0;
            current_angular_correction[1] = 0.0;
            current_angular_correction[2] = 0.0;

            D_upper[0] =
                D_calculator.X[0] * D_calculator.X[0] * D_calculator.d3 + 2.0 * D_calculator.d2;
            m2m_vector d3_X00 = D_calculator.d3 * D_calculator.X_00;
            D_upper[0] += D_calculator.d2;
            D_upper[0] += 5.0 * d3_X00;
            m2m_vector d3_X01 = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];
            D_upper[1] = 3.0 * d3_X01;
            m2m_vector X_02 = D_calculator.X[0] * D_calculator.X[2];
            m2m_vector d3_X02 = D_calculator.d3 * X_02;
            D_upper[2] = 3.0 * d3_X02;
            m2m_vector n0_tmp = m_partner[10] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[0] * factor_sixth_v[10]);
            current_angular_correction[1] -= n0_tmp * (D_upper[1] * factor_sixth_v[10]);
            current_angular_correction[2] -= n0_tmp * (D_upper[2] * factor_sixth_v[10]);

            D_upper[3] = D_calculator.d2;
            m2m_vector d3_X11 = D_calculator.d3 * D_calculator.X_11;
            D_upper[3] += d3_X11;
            D_upper[3] += D_calculator.d3 * D_calculator.X_00;
            m2m_vector d3_X12 = D_calculator.d3 * D_calculator.X[1] * D_calculator.X[2];
            D_upper[4] = d3_X12;

            n0_tmp = m_partner[11] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[1] * factor_sixth_v[11]);
            current_angular_correction[1] -= n0_tmp * (D_upper[3] * factor_sixth_v[11]);
            current_angular_correction[2] -= n0_tmp * (D_upper[4] * factor_sixth_v[11]);

            D_upper[5] = D_calculator.d2;
            m2m_vector d3_X22 = D_calculator.d3 * D_calculator.X_22;
            D_upper[5] += d3_X22;
            D_upper[5] += d3_X00;

            n0_tmp = m_partner[12] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[2] * factor_sixth_v[12]);
            current_angular_correction[1] -= n0_tmp * (D_upper[4] * factor_sixth_v[12]);
            current_angular_correction[2] -= n0_tmp * (D_upper[5] * factor_sixth_v[12]);

            D_upper[6] = 3.0 * d3_X01;
            D_upper[7] = D_calculator.d3 * X_02;

            n0_tmp = m_partner[13] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[3] * factor_sixth_v[13]);
            current_angular_correction[1] -= n0_tmp * (D_upper[6] * factor_sixth_v[13]);
            current_angular_correction[2] -= n0_tmp * (D_upper[7] * factor_sixth_v[13]);

            D_upper[8] = D_calculator.d3 * D_calculator.X[0] * D_calculator.X[1];

            n0_tmp = m_partner[14] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[4] * factor_sixth_v[14]);
            current_angular_correction[1] -= n0_tmp * (D_upper[7] * factor_sixth_v[14]);
            current_angular_correction[2] -= n0_tmp * (D_upper[8] * factor_sixth_v[14]);

            D_upper[9] = 3.0 * d3_X02;

            n0_tmp = m_partner[15] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[5] * factor_sixth_v[15]);
            current_angular_correction[1] -= n0_tmp * (D_upper[8] * factor_sixth_v[15]);
            current_angular_correction[2] -= n0_tmp * (D_upper[9] * factor_sixth_v[15]);

            D_upper[10] =
                D_calculator.X[1] * D_calculator.X[1] * D_calculator.d3 + 2.0 * D_calculator.d2;
            D_upper[10] += D_calculator.d2;
            D_upper[10] += 5.0 * d3_X11;

            D_upper[11] = 3.0 * d3_X12;

            n0_tmp = m_partner[16] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[6] * factor_sixth_v[16]);
            current_angular_correction[1] -= n0_tmp * (D_upper[10] * factor_sixth_v[16]);
            current_angular_correction[2] -= n0_tmp * (D_upper[11] * factor_sixth_v[16]);

            D_upper[12] = D_calculator.d2;
            D_upper[12] += d3_X22;
            D_upper[12] += d3_X11;

            n0_tmp = m_partner[17] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[7] * factor_sixth_v[17]);
            current_angular_correction[1] -= n0_tmp * (D_upper[11] * factor_sixth_v[17]);
            current_angular_correction[2] -= n0_tmp * (D_upper[12] * factor_sixth_v[17]);

            D_upper[13] = 3.0 * d3_X12;

            n0_tmp = m_partner[18] - m_cell_iterator.value() * n0_constant;
            m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[8] * factor_sixth_v[18]);
            current_angular_correction[1] -= n0_tmp * (D_upper[12] * factor_sixth_v[18]);
            current_angular_correction[2] -= n0_tmp * (D_upper[13] * factor_sixth_v[18]);

            D_upper[14] =
                D_calculator.X[2] * D_calculator.X[2] * D_calculator.d3 + 2.0 * D_calculator.d2;
            D_upper[14] += D_calculator.d2;
            D_upper[14] += 5.0 * d3_X22;

            n0_tmp = m_partner[19] - m_cell_iterator.value() * n0_constant;
            // m_cell_iterator++;

            current_angular_correction[0] -= n0_tmp * (D_upper[9] * factor_sixth_v[19]);
            current_angular_correction[1] -= n0_tmp * (D_upper[13] * factor_sixth_v[19]);
            current_angular_correction[2] -= n0_tmp * (D_upper[14] * factor_sixth_v[19]);

            tmp = current_angular_correction_result.value() + current_angular_correction[0];
            Vc::where(mask, tmp).memstore(
                current_angular_correction_result.pointer(), Vc::flags::element_aligned);
            current_angular_correction_result++;
            tmp = current_angular_correction_result.value() + current_angular_correction[1];
            Vc::where(mask, tmp).memstore(
                current_angular_correction_result.pointer(), Vc::flags::element_aligned);
            current_angular_correction_result++;
            tmp = current_angular_correction_result.value() + current_angular_correction[2];
            Vc::where(mask, tmp).memstore(
                current_angular_correction_result.pointer(), Vc::flags::element_aligned);
        }
    }

    void m2m_kernel::blocked_interaction_non_rho(const multiindex<>& cell_index,
        const int64_t cell_flat_index, const multiindex<m2m_int_vector>& cell_index_coarse,
        const multiindex<>& cell_index_unpadded, const int64_t cell_flat_index_unpadded,
        const std::vector<multiindex<>>& stencil, const size_t outer_stencil_index) {
        // TODO: should change name to something better (not taylor, but space_vector)
        // struct_of_array_taylor<space_vector, real, 3> X =
        //     center_of_masses_SoA.get_view(cell_flat_index);

        struct_of_array_iterator<space_vector, real, 3> X(center_of_masses_SoA, cell_flat_index);
        for (size_t inner_stencil_index = 0; inner_stencil_index < STENCIL_BLOCKING &&
             outer_stencil_index + inner_stencil_index < stencil.size();
             inner_stencil_index += 1) {
            const multiindex<>& stencil_element =
                stencil[outer_stencil_index + inner_stencil_index];
            const multiindex<> interaction_partner_index(cell_index.x + stencil_element.x,
                cell_index.y + stencil_element.y, cell_index.z + stencil_element.z);

            const int64_t interaction_partner_flat_index =
                to_flat_index_padded(interaction_partner_index);

            // check whether all vector elements are in empty border
            if (vector_is_empty[interaction_partner_flat_index]) {
                continue;
            }

            // implicitly broadcasts to vector
            multiindex<m2m_int_vector> interaction_partner_index_coarse(interaction_partner_index);
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

            struct_of_array_iterator<space_vector, real, 3> Y(
                center_of_masses_SoA, interaction_partner_flat_index);

            // distance between cells in all dimensions
            // TODO: replace by m2m_vector for vectorization or get rid of temporary
            m2m_vector dX[NDIM];
            dX[0] = X.value() - Y.value();
            X++;
            Y++;
            dX[1] = X.value() - Y.value();
            X++;
            Y++;
            dX[2] = X.value() - Y.value();

            // reset X for next iteration
            X.decrement(2);

            expansion_v m_partner;
            {
                struct_of_array_iterator<expansion, real, 20> m_partner_iterator(
                    local_expansions_SoA, interaction_partner_flat_index);
                m_partner[0] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[1] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[2] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[3] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[4] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[5] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[6] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[7] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[8] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[9] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[10] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[11] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[12] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[13] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[14] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[15] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[16] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[17] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[18] = m_partner_iterator.value();
                m_partner_iterator++;
                m_partner[19] = m_partner_iterator.value();
                m_partner_iterator++;
            }

            // R_i in paper is the dX in the code
            // D is taylor expansion value for a given X expansion of the gravitational
            // potential
            // (multipole expansion)

            // calculates all D-values, calculate all coefficients of 1/r (not the
            // potential),
            // formula (6)-(9) and (19)
            D_split D_calculator(dX);
            m2m_vector D_lower[20];
            D_calculator.calculate_D_lower(D_lower);

            // expansion_v current_potential;
            m2m_vector cur_pot[10];

            // 10-19 are not cached!

            // the following loops calculate formula (10), potential from B->A
            cur_pot[0] = m_partner[0] * D_lower[0];
            cur_pot[1] = m_partner[0] * D_lower[1];
            cur_pot[2] = m_partner[0] * D_lower[2];
            cur_pot[3] = m_partner[0] * D_lower[3];

            cur_pot[0] -= m_partner[1] * D_lower[1];
            cur_pot[1] -= m_partner[1] * D_lower[4];
            cur_pot[1] -= m_partner[1] * D_lower[5];
            cur_pot[1] -= m_partner[1] * D_lower[6];

            cur_pot[0] -= m_partner[2] * D_lower[2];
            cur_pot[2] -= m_partner[2] * D_lower[5];
            cur_pot[2] -= m_partner[2] * D_lower[7];
            cur_pot[2] -= m_partner[2] * D_lower[8];

            cur_pot[0] -= m_partner[3] * D_lower[3];
            cur_pot[3] -= m_partner[3] * D_lower[6];
            cur_pot[3] -= m_partner[3] * D_lower[8];
            cur_pot[3] -= m_partner[3] * D_lower[9];

            cur_pot[0] += m_partner[4] * (D_lower[4] * factor_half_v[4]);
            cur_pot[1] += m_partner[4] * (D_lower[10] * factor_half_v[4]);
            cur_pot[2] += m_partner[4] * (D_lower[11] * factor_half_v[4]);
            cur_pot[3] += m_partner[4] * (D_lower[12] * factor_half_v[4]);

            cur_pot[0] += m_partner[5] * (D_lower[5] * factor_half_v[5]);
            cur_pot[1] += m_partner[5] * (D_lower[11] * factor_half_v[5]);
            cur_pot[2] += m_partner[5] * (D_lower[13] * factor_half_v[5]);
            cur_pot[3] += m_partner[5] * (D_lower[14] * factor_half_v[5]);

            cur_pot[0] += m_partner[6] * (D_lower[6] * factor_half_v[6]);
            cur_pot[1] += m_partner[6] * (D_lower[12] * factor_half_v[6]);
            cur_pot[2] += m_partner[6] * (D_lower[14] * factor_half_v[6]);
            cur_pot[3] += m_partner[6] * (D_lower[15] * factor_half_v[6]);

            cur_pot[0] += m_partner[7] * (D_lower[7] * factor_half_v[7]);
            cur_pot[1] += m_partner[7] * (D_lower[13] * factor_half_v[7]);
            cur_pot[2] += m_partner[7] * (D_lower[16] * factor_half_v[7]);
            cur_pot[3] += m_partner[7] * (D_lower[17] * factor_half_v[7]);

            cur_pot[0] += m_partner[8] * (D_lower[8] * factor_half_v[8]);
            cur_pot[1] += m_partner[8] * (D_lower[14] * factor_half_v[8]);
            cur_pot[2] += m_partner[8] * (D_lower[17] * factor_half_v[8]);
            cur_pot[3] += m_partner[8] * (D_lower[18] * factor_half_v[8]);

            cur_pot[0] += m_partner[9] * (D_lower[9] * factor_half_v[9]);
            cur_pot[1] += m_partner[9] * (D_lower[15] * factor_half_v[9]);
            cur_pot[2] += m_partner[9] * (D_lower[18] * factor_half_v[9]);
            cur_pot[3] += m_partner[9] * (D_lower[19] * factor_half_v[9]);

            cur_pot[0] -= m_partner[10] * (D_lower[10] * factor_sixth_v[10]);
            cur_pot[0] -= m_partner[11] * (D_lower[11] * factor_sixth_v[11]);
            cur_pot[0] -= m_partner[12] * (D_lower[12] * factor_sixth_v[12]);
            cur_pot[0] -= m_partner[13] * (D_lower[13] * factor_sixth_v[13]);
            cur_pot[0] -= m_partner[14] * (D_lower[14] * factor_sixth_v[14]);
            cur_pot[0] -= m_partner[15] * (D_lower[15] * factor_sixth_v[15]);
            cur_pot[0] -= m_partner[16] * (D_lower[16] * factor_sixth_v[16]);
            cur_pot[0] -= m_partner[17] * (D_lower[17] * factor_sixth_v[17]);
            cur_pot[0] -= m_partner[18] * (D_lower[18] * factor_sixth_v[18]);
            cur_pot[0] -= m_partner[19] * (D_lower[19] * factor_sixth_v[19]);

            cur_pot[4] = m_partner[0] * D_lower[4];
            cur_pot[5] = m_partner[0] * D_lower[5];
            cur_pot[6] = m_partner[0] * D_lower[6];
            cur_pot[7] = m_partner[0] * D_lower[7];
            cur_pot[8] = m_partner[0] * D_lower[8];
            cur_pot[9] = m_partner[0] * D_lower[9];

            cur_pot[4] -= m_partner[1] * D_lower[10];
            cur_pot[5] -= m_partner[1] * D_lower[11];
            cur_pot[6] -= m_partner[1] * D_lower[12];
            cur_pot[7] -= m_partner[1] * D_lower[13];
            cur_pot[8] -= m_partner[1] * D_lower[14];
            cur_pot[9] -= m_partner[1] * D_lower[15];

            cur_pot[4] -= m_partner[2] * D_lower[11];
            cur_pot[5] -= m_partner[2] * D_lower[13];
            cur_pot[6] -= m_partner[2] * D_lower[14];
            cur_pot[7] -= m_partner[2] * D_lower[16];
            cur_pot[8] -= m_partner[2] * D_lower[17];
            cur_pot[9] -= m_partner[2] * D_lower[18];

            cur_pot[4] -= m_partner[3] * D_lower[12];
            cur_pot[5] -= m_partner[3] * D_lower[14];
            cur_pot[6] -= m_partner[3] * D_lower[15];
            cur_pot[7] -= m_partner[3] * D_lower[17];
            cur_pot[8] -= m_partner[3] * D_lower[18];
            cur_pot[9] -= m_partner[3] * D_lower[19];

            struct_of_array_iterator<expansion, real, 20> current_potential_result(
                potential_expansions_SoA, cell_flat_index_unpadded);

            m2m_vector tmp = current_potential_result.value() + cur_pot[0];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;

            tmp = current_potential_result.value() + cur_pot[1];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[2];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[3];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;

            tmp = current_potential_result.value() + cur_pot[4];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[5];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[6];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[7];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[8];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + cur_pot[9];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;    // don't remove this

            tmp = current_potential_result.value() + m_partner[0] * D_lower[10];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[11];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[12];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[13];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[14];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[15];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[16];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[17];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[18];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D_lower[19];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
        }
    }
}    // namespace fmm
}    // namespace octotiger
