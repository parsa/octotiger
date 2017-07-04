#include "m2m_kernel.hpp"

#include "grid_flattened_indices.hpp"
#include "helper.hpp"
#include "m2m_taylor_set_basis.hpp"
#include "struct_of_array_taylor.hpp"

extern taylor<4, real> factor;

namespace octotiger {
namespace fmm {

    // TODO:
    // - check codegen and fix in Vc
    // - check for amount of temporaries
    // - try to replace expensive operations like sqrt
    // - remove all sqr()
    // - increase INX

    void m2m_kernel::blocked_interaction(const multiindex<>& cell_index,
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

            // // check whether all vector elements are in empty border
            // const multiindex<> in_boundary_start(
            //     (interaction_partner_index.x / INNER_CELLS_PER_DIRECTION) - 1,
            //     (interaction_partner_index.y / INNER_CELLS_PER_DIRECTION) - 1,
            //     (interaction_partner_index.z / INNER_CELLS_PER_DIRECTION) - 1);

            // const multiindex<> in_boundary_end(
            //     (interaction_partner_index.x / INNER_CELLS_PER_DIRECTION) - 1,
            //     (interaction_partner_index.y / INNER_CELLS_PER_DIRECTION) - 1,
            //     ((interaction_partner_index.z + m2m_int_vector::size()) /
            //         INNER_CELLS_PER_DIRECTION) -
            //         1);

            // geo::direction dir_start;
            // dir_start.set(in_boundary_start.x, in_boundary_start.y, in_boundary_start.z);
            // geo::direction dir_end;
            // dir_end.set(in_boundary_end.x, in_boundary_end.y, in_boundary_end.z);

            const int64_t interaction_partner_flat_index =
                to_flat_index_padded(interaction_partner_index);

            // if (neighbor_empty[dir_start.flat_index_with_center()] &&
            //     neighbor_empty[dir_end.flat_index_with_center()]) {
            //     if (!vector_is_empty[interaction_partner_flat_index]) {
            //         std::cout << "not true, but should be" << std::endl;
            //         // std::cout << "interaction_partner_index:" << interaction_partner_index
            //         //           << std::endl;
            //         // std::cout << "interaction_partner_flat_index: "
            //         //           << interaction_partner_flat_index << std::endl;
            //         // std::cout << "dir_start.flat_index_with_center(): "
            //         //           << dir_start.flat_index_with_center() << std::endl;
            //         // std::cout << "dir_end.flat_index_with_center(): "
            //         //           << dir_end.flat_index_with_center() << std::endl;
            // 	    // std::cout << "in_boundary_end: " << in_boundary_end << std::endl;
            //     }
            //     continue;
            // } else {
            //     if (vector_is_empty[interaction_partner_flat_index]) {
            //         std::cout << "expected false, got true" << std::endl;
            //         // std::cout << "dir_start.flat_index_with_center(): "
            //         //           << dir_start.flat_index_with_center() << std::endl;
            //         // std::cout << "dir_end.flat_index_with_center(): "
            //         //           << dir_end.flat_index_with_center() << std::endl;
            //     }
            // }

            if (vector_is_empty[interaction_partner_flat_index]) {
                continue;
            }

            // implicitly broadcasts to vector
            multiindex<m2m_int_vector> interaction_partner_index_coarse(interaction_partner_index);
            interaction_partner_index_coarse.z += offset_vector;
            // note that this is the same for groups of 2x2x2 elements
            // -> maps to the same for some SIMD lanes
            interaction_partner_index_coarse.transform_coarse();
            // m2m_int_vector c_x;
            // c_x.memload(&coarse_x[interaction_partner_flat_index], Vc::flags::element_aligned);
            // m2m_int_vector c_y;
            // c_y.memload(&coarse_y[interaction_partner_flat_index], Vc::flags::element_aligned);
            // m2m_int_vector c_z;
            // c_z.memload(&coarse_z[interaction_partner_flat_index], Vc::flags::element_aligned);
            // multiindex<m2m_int_vector> interaction_partner_index_coarse(c_x, c_y, c_z);

            m2m_int_vector theta_c_rec_squared_int = detail::distance_squared_reciprocal(
                cell_index_coarse, interaction_partner_index_coarse);

            m2m_vector theta_c_rec_squared =
                // Vc::static_datapar_cast<double>(theta_c_rec_squared_int);
                Vc::static_datapar_cast_double_to_int(theta_c_rec_squared_int);
            // TODO: use static_datapar_cast after issue is fixed
            // m2m_vector theta_c_rec_squared;
            // for (size_t i = 0; i < m2m_vector::size(); i++) {
            //     theta_c_rec_squared[i] = static_cast<double>(theta_c_rec_squared_int[i]);
            // }

            m2m_vector::mask_type mask = theta_rec_squared > theta_c_rec_squared;

            if (Vc::none_of(mask)) {
                continue;
            }

            // struct_of_array_taylor<space_vector, real, 3> Y =
            //     center_of_masses_SoA.get_view(interaction_partner_flat_index);
            struct_of_array_iterator<space_vector, real, 3> Y(
                center_of_masses_SoA, interaction_partner_flat_index);

            // distance between cells in all dimensions
            // TODO: replace by m2m_vector for vectorization or get rid of temporary
            std::array<m2m_vector, NDIM> dX;
            // for (integer d = 0; d < NDIM; ++d) {
            //     dX[d] = X.component(d) - Y.component(d);
            // }
            // dX[0] = X.component(0) - Y.component(0);
            // dX[1] = X.component(1) - Y.component(1);
            // dX[2] = X.component(2) - Y.component(2);
            // dX[0] = m2m_vector(*X, Vc::flags::element_aligned) -
            //     m2m_vector(*Y, Vc::flags::element_aligned);
            dX[0] = X.value() - Y.value();
            X++;
            Y++;
            // dX[1] = m2m_vector(*X, Vc::flags::element_aligned) -
            //     m2m_vector(*Y, Vc::flags::element_aligned);
            dX[1] = X.value() - Y.value();
            X++;
            Y++;
            // dX[2] = m2m_vector(*X, Vc::flags::element_aligned) -
            //     m2m_vector(*Y, Vc::flags::element_aligned);
            dX[2] = X.value() - Y.value();

            // reset X for next iteration
            X.decrement(2);

            // struct_of_array_taylor<expansion, real, 20> m_partner_view =
            //     local_expansions_SoA.get_view(interaction_partner_flat_index);

            expansion_v m_partner;
            // for (size_t i = 0; i < m_partner.size(); i++) {
            //     // m_partner[i] = m_partner_view.component(i);
            //     m_partner[i] = m2m_vector(*m_partner_iterator, Vc::flags::element_aligned);
            //     m_partner_iterator++;
            // }

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

            // m_partner[0] = m_partner_view.component(0);
            // m_partner[1] = m_partner_view.component(1);
            // m_partner[2] = m_partner_view.component(2);
            // m_partner[3] = m_partner_view.component(3);
            // m_partner[4] = m_partner_view.component(4);
            // m_partner[5] = m_partner_view.component(5);
            // m_partner[6] = m_partner_view.component(6);
            // m_partner[7] = m_partner_view.component(7);
            // m_partner[8] = m_partner_view.component(8);
            // m_partner[9] = m_partner_view.component(9);
            // m_partner[10] = m_partner_view.component(10);
            // m_partner[11] = m_partner_view.component(11);
            // m_partner[12] = m_partner_view.component(12);
            // m_partner[13] = m_partner_view.component(13);
            // m_partner[14] = m_partner_view.component(14);
            // m_partner[15] = m_partner_view.component(15);
            // m_partner[16] = m_partner_view.component(16);
            // m_partner[17] = m_partner_view.component(17);
            // m_partner[18] = m_partner_view.component(18);
            // m_partner[19] = m_partner_view.component(19);

            // m2m_vector grad_0 = m_partner.grad_0();
            // m2m_vector grad_10 = m_partner.grad_1(0);
            // m2m_vector grad_11 = m_partner.grad_1(1);
            // m2m_vector grad_12 = m_partner.grad_1(2);

            // TODO: replace by reference to get rid of temporary?
            taylor<4, m2m_vector> n0;

            // Initalize moments and momentum
            if (type != RHO) {
                for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                    n0[j] = ZERO;
                }
            } else {
                // this branch computes the angular momentum correction, (20) in the
                // paper divide by mass of other cell
                // struct_of_array_taylor<expansion, real, 20> m_cell =
                //     local_expansions_SoA.get_view(cell_flat_index);
                struct_of_array_iterator<expansion, real, 20> m_cell_iterator(
                    local_expansions_SoA, cell_flat_index);

                // m2m_vector const tmp1 = m_partner[0] / m_cell.grad_0();
                m2m_vector const tmp1 = m_partner[0] / m_cell_iterator.value();

                m_cell_iterator.increment(10);

                // calculating the coefficients for formula (M are the octopole moments)
                // the coefficients are calculated in (17) and (18)
                // for (integer j = taylor_sizes[2]; j != taylor_sizes[3]; ++j) {
                //     // n0[j] = m_partner[j] - m_cell.component(j) * tmp1;
                //     n0[j] = m_partner[j] -
                //         m_cell_iterator.value() * tmp1;
                //     m_cell_iterator++;
                // }
                n0[10] = m_partner[10] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[11] = m_partner[11] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[12] = m_partner[12] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[13] = m_partner[13] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[14] = m_partner[14] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[15] = m_partner[15] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[16] = m_partner[16] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[17] = m_partner[17] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[18] = m_partner[18] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
                n0[19] = m_partner[19] - m_cell_iterator.value() * tmp1;
                m_cell_iterator++;
            }

            // R_i in paper is the dX in the code
            // D is taylor expansion value for a given X expansion of the gravitational
            // potential
            // (multipole expansion)

            // calculates all D-values, calculate all coefficients of 1/r (not the
            // potential),
            // formula (6)-(9) and (19)
            // D.set_basis(dX);
            taylor<5, m2m_vector> D;
            octotiger::fmm::set_basis(D, dX);

            // output variable references
            // TODO: use these again after debugging individual components!
            // expansion& current_potential =
            // potential_expansions[cell_flat_index_unpadded];
            // space_vector& current_angular_correction =
            // angular_corrections[cell_flat_index_unpadded];

            // expansion_v current_potential;
            m2m_vector cur_pot[9];
            // for (size_t i = 0; i < current_potential.size(); i++) {
            //     current_potential[i] = 0.0;
            // }
            // current_potential[0] = 0.0;
            // current_potential[1] = 0.0;
            // current_potential[2] = 0.0;
            // current_potential[3] = 0.0;
            // current_potential[4] = 0.0;
            // current_potential[5] = 0.0;
            // current_potential[6] = 0.0;
            // current_potential[7] = 0.0;
            // current_potential[8] = 0.0;
            // current_potential[9] = 0.0;
            cur_pot[0] = 0.0;
            cur_pot[1] = 0.0;
            cur_pot[2] = 0.0;
            cur_pot[3] = 0.0;
            cur_pot[4] = 0.0;
            cur_pot[5] = 0.0;
            cur_pot[6] = 0.0;
            cur_pot[7] = 0.0;
            cur_pot[8] = 0.0;

            // 10-19 not cached
            // current_potential[10] = 0.0;
            // current_potential[11] = 0.0;
            // current_potential[12] = 0.0;
            // current_potential[13] = 0.0;
            // current_potential[14] = 0.0;
            // current_potential[15] = 0.0;
            // current_potential[16] = 0.0;
            // current_potential[17] = 0.0;
            // current_potential[18] = 0.0;
            // current_potential[19] = 0.0;

            // the following loops calculate formula (10), potential from B->A
            m2m_vector p_0 = m_partner[0] * D[0];
            // current_potential[0] += m_partner.component(0) * D[0];
            // m2m_vector tmp_c0 =
            //     current_potential_result.component(0) + m_partner.component(0) * D[0];

            if (type != RHO) {
                // for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
                //     current_potential[0] -= m_partner.component(i) * D[i];
                //     // tmp_c0 -= m_partner.component(i) * D[i];
                // }
                // current_potential[0] -= m_partner.component(1) * D[1];
                // current_potential[0] -= m_partner.component(2) * D[2];
                // current_potential[0] -= m_partner.component(3) * D[3];

                p_0 -= m_partner[1] * D[1];
                p_0 -= m_partner[2] * D[2];
                p_0 -= m_partner[3] * D[3];
            }

            // for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
            //     const auto tmp = D[i] * (factor[i] * HALF);
            //     current_potential[0] += m_partner.component(i) * tmp;
            //     // tmp_c0 += m_partner.component(i) * tmp;
            // }
            // current_potential[0] += m_partner.component(4) * (D[4] * (factor[4] * HALF));
            // current_potential[0] += m_partner.component(5) * (D[5] * (factor[5] * HALF));
            // current_potential[0] += m_partner.component(6) * (D[6] * (factor[6] * HALF));
            // current_potential[0] += m_partner.component(7) * (D[7] * (factor[7] * HALF));
            // current_potential[0] += m_partner.component(8) * (D[8] * (factor[8] * HALF));
            // current_potential[0] += m_partner.component(9) * (D[9] * (factor[9] * HALF));

            p_0 += m_partner[4] * (D[4] * (factor[4] * HALF));
            p_0 += m_partner[5] * (D[5] * (factor[5] * HALF));
            p_0 += m_partner[6] * (D[6] * (factor[6] * HALF));
            p_0 += m_partner[7] * (D[7] * (factor[7] * HALF));
            p_0 += m_partner[8] * (D[8] * (factor[8] * HALF));
            p_0 += m_partner[9] * (D[9] * (factor[9] * HALF));

            // for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
            //     const auto tmp = D[i] * (factor[i] * SIXTH);
            //     current_potential[0] -= m_partner.component(i) * tmp;
            //     // tmp_c0 -= m_partner.component(i) * tmp;
            // }

            p_0 -= m_partner[10] * (D[10] * (factor[10] * SIXTH));
            p_0 -= m_partner[11] * (D[11] * (factor[11] * SIXTH));
            p_0 -= m_partner[12] * (D[12] * (factor[12] * SIXTH));
            p_0 -= m_partner[13] * (D[13] * (factor[13] * SIXTH));
            p_0 -= m_partner[14] * (D[14] * (factor[14] * SIXTH));
            p_0 -= m_partner[15] * (D[15] * (factor[15] * SIXTH));
            p_0 -= m_partner[16] * (D[16] * (factor[16] * SIXTH));
            p_0 -= m_partner[17] * (D[17] * (factor[17] * SIXTH));
            p_0 -= m_partner[18] * (D[18] * (factor[18] * SIXTH));
            p_0 -= m_partner[19] * (D[19] * (factor[19] * SIXTH));

            // current_potential[0] = p_0;
            struct_of_array_iterator<expansion, real, 20> current_potential_result(
                potential_expansions_SoA, cell_flat_index_unpadded);

            m2m_vector tmp = current_potential_result.value() + p_0;
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;

            // Vc::where(mask, tmp_c0)
            //     .memstore(
            //         current_potential_result.component_pointer(0), Vc::flags::element_aligned);

            if (type != RHO) {
                // for (integer a = 0; a < NDIM; ++a) {
                //     int const* ab_idx_map = to_ab_idx_map3[a];
                //     for (integer i = 0; i < 3; ++i) {
                //         current_potential(a) -= m_partner.grad_1(a) * D[ab_idx_map[i]];
                //     }
                // } //m_partner.grad_1(0);
                // current_potential(0) -= m_partner[1] * D[4];
                // current_potential(0) -= m_partner[1] * D[5];
                // current_potential(0) -= m_partner[1] * D[6];

                // current_potential(1) -= m_partner[2] * D[5];
                // current_potential(1) -= m_partner[2] * D[7];
                // current_potential(1) -= m_partner[2] * D[8];

                // current_potential(2) -= m_partner[3] * D[6];
                // current_potential(2) -= m_partner[3] * D[8];
                // current_potential(2) -= m_partner[3] * D[9];

                cur_pot[0] -= m_partner[1] * D[4];
                cur_pot[0] -= m_partner[1] * D[5];
                cur_pot[0] -= m_partner[1] * D[6];

                cur_pot[1] -= m_partner[2] * D[5];
                cur_pot[1] -= m_partner[2] * D[7];
                cur_pot[1] -= m_partner[2] * D[8];

                cur_pot[2] -= m_partner[3] * D[6];
                cur_pot[2] -= m_partner[3] * D[8];
                cur_pot[2] -= m_partner[3] * D[9];
            }

            // for (integer a = 0; a < NDIM; ++a) {
            //     int const* abc_idx_map = to_abc_idx_map3[a];
            //     current_potential(a) += grad_0 * D(a);
            //     for (integer i = 0; i != 6; ++i) {
            //         const integer cb_idx = cb_idx_map[i];
            //         const auto tmp1 = D[abc_idx_map[i]] * (factor[cb_idx] * HALF);
            //         current_potential(a) += m_partner.component(cb_idx) * tmp1;
            //     }
            // }

            // current_potential(0) += m_partner[0] * D(0);
            // current_potential(1) += m_partner[0] * D(1);
            // current_potential(2) += m_partner[0] * D(2);

            cur_pot[0] += m_partner[0] * D(0);
            cur_pot[1] += m_partner[0] * D(1);
            cur_pot[2] += m_partner[0] * D(2);

            // current_potential(0) += m_partner[4] * (D[10] * (factor[4] * HALF));
            // current_potential(0) += m_partner[5] * (D[11] * (factor[5] * HALF));
            // current_potential(0) += m_partner[6] * (D[12] * (factor[6] * HALF));
            // current_potential(0) += m_partner[7] * (D[13] * (factor[7] * HALF));
            // current_potential(0) += m_partner[8] * (D[14] * (factor[8] * HALF));
            // current_potential(0) += m_partner[9] * (D[15] * (factor[9] * HALF));

            cur_pot[0] += m_partner[4] * (D[10] * (factor[4] * HALF));
            cur_pot[0] += m_partner[5] * (D[11] * (factor[5] * HALF));
            cur_pot[0] += m_partner[6] * (D[12] * (factor[6] * HALF));
            cur_pot[0] += m_partner[7] * (D[13] * (factor[7] * HALF));
            cur_pot[0] += m_partner[8] * (D[14] * (factor[8] * HALF));
            cur_pot[0] += m_partner[9] * (D[15] * (factor[9] * HALF));

            // current_potential(1) += m_partner[4] * (D[11] * (factor[4] * HALF));
            // current_potential(1) += m_partner[5] * (D[13] * (factor[5] * HALF));
            // current_potential(1) += m_partner[6] * (D[14] * (factor[6] * HALF));
            // current_potential(1) += m_partner[7] * (D[16] * (factor[7] * HALF));
            // current_potential(1) += m_partner[8] * (D[17] * (factor[8] * HALF));
            // current_potential(1) += m_partner[9] * (D[18] * (factor[9] * HALF));

            cur_pot[1] += m_partner[4] * (D[11] * (factor[4] * HALF));
            cur_pot[1] += m_partner[5] * (D[13] * (factor[5] * HALF));
            cur_pot[1] += m_partner[6] * (D[14] * (factor[6] * HALF));
            cur_pot[1] += m_partner[7] * (D[16] * (factor[7] * HALF));
            cur_pot[1] += m_partner[8] * (D[17] * (factor[8] * HALF));
            cur_pot[1] += m_partner[9] * (D[18] * (factor[9] * HALF));

            // current_potential(2) += m_partner[4] * (D[12] * (factor[4] * HALF));
            // current_potential(2) += m_partner[5] * (D[14] * (factor[5] * HALF));
            // current_potential(2) += m_partner[6] * (D[15] * (factor[6] * HALF));
            // current_potential(2) += m_partner[7] * (D[17] * (factor[7] * HALF));
            // current_potential(2) += m_partner[8] * (D[18] * (factor[8] * HALF));
            // current_potential(2) += m_partner[9] * (D[19] * (factor[9] * HALF));

            cur_pot[2] += m_partner[4] * (D[12] * (factor[4] * HALF));
            cur_pot[2] += m_partner[5] * (D[14] * (factor[5] * HALF));
            cur_pot[2] += m_partner[6] * (D[15] * (factor[6] * HALF));
            cur_pot[2] += m_partner[7] * (D[17] * (factor[7] * HALF));
            cur_pot[2] += m_partner[8] * (D[18] * (factor[8] * HALF));
            cur_pot[2] += m_partner[9] * (D[19] * (factor[9] * HALF));

            tmp = current_potential_result.value() + cur_pot[0];
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

            // integer const ab_idx = {4,  5,  6,  7,  8,  9,};
            // for (integer i = 0; i != 6; ++i) {
            //     current_potential[ab_idx] += grad_0 * D[ab_idx];
            // }

            // current_potential[4] += m_partner[0] * D[4];
            // current_potential[5] += m_partner[0] * D[5];
            // current_potential[6] += m_partner[0] * D[6];
            // current_potential[7] += m_partner[0] * D[7];
            // current_potential[8] += m_partner[0] * D[8];
            // current_potential[9] += m_partner[0] * D[9];

            cur_pot[3] += m_partner[0] * D[4];
            cur_pot[4] += m_partner[0] * D[5];
            cur_pot[5] += m_partner[0] * D[6];
            cur_pot[6] += m_partner[0] * D[7];
            cur_pot[7] += m_partner[0] * D[8];
            cur_pot[8] += m_partner[0] * D[9];

            // int const* abc_idx_map6 = to_abc_idx_map6[i];
            // integer const ab_idx = {4, 5, 6, 7, 8, 9};
            // for (integer i = 0; i != 6; ++i) {
            //     for (integer c = 0; c < NDIM; ++c) {
            //         current_potential[ab_idx] -= m_partner.grad_1(c) * D[abc_idx_map6[c]];
            //     }
            // }

            // current_potential[4] -= m_partner[1] * D[10];
            // current_potential[4] -= m_partner[2] * D[11];
            // current_potential[4] -= m_partner[3] * D[12];

            // current_potential[5] -= m_partner[1] * D[11];
            // current_potential[5] -= m_partner[2] * D[13];
            // current_potential[5] -= m_partner[3] * D[14];

            // current_potential[6] -= m_partner[1] * D[12];
            // current_potential[6] -= m_partner[2] * D[14];
            // current_potential[6] -= m_partner[3] * D[15];

            // current_potential[7] -= m_partner[1] * D[13];
            // current_potential[7] -= m_partner[2] * D[16];
            // current_potential[7] -= m_partner[3] * D[17];

            // current_potential[8] -= m_partner[1] * D[14];
            // current_potential[8] -= m_partner[2] * D[17];
            // current_potential[8] -= m_partner[3] * D[18];

            // current_potential[9] -= m_partner[1] * D[15];
            // current_potential[9] -= m_partner[2] * D[18];
            // current_potential[9] -= m_partner[3] * D[19];

            cur_pot[3] -= m_partner[1] * D[10];
            cur_pot[3] -= m_partner[2] * D[11];
            cur_pot[3] -= m_partner[3] * D[12];

            cur_pot[4] -= m_partner[1] * D[11];
            cur_pot[4] -= m_partner[2] * D[13];
            cur_pot[4] -= m_partner[3] * D[14];

            cur_pot[5] -= m_partner[1] * D[12];
            cur_pot[5] -= m_partner[2] * D[14];
            cur_pot[5] -= m_partner[3] * D[15];

            cur_pot[6] -= m_partner[1] * D[13];
            cur_pot[6] -= m_partner[2] * D[16];
            cur_pot[6] -= m_partner[3] * D[17];

            cur_pot[7] -= m_partner[1] * D[14];
            cur_pot[7] -= m_partner[2] * D[17];
            cur_pot[7] -= m_partner[3] * D[18];

            cur_pot[8] -= m_partner[1] * D[15];
            cur_pot[8] -= m_partner[2] * D[18];
            cur_pot[8] -= m_partner[3] * D[19];

            // struct_of_array_iterator<expansion, real, 20> current_potential_result(
            //     potential_expansions_SoA, cell_flat_index_unpadded);

            // for (size_t i = 0; i < 10; i++) {
            //     // m2m_vector tmp = current_potential_result.component(i) + current_potential[i];
            //     m2m_vector tmp = m2m_vector(*current_potential_result,
            //     Vc::flags::element_aligned) +
            //         current_potential[i];
            //     // Vc::where(mask, tmp).memstore(
            //     //     current_potential_result.component_pointer(i),
            //     Vc::flags::element_aligned);
            //     Vc::where(mask, tmp).memstore(
            //         *current_potential_result, Vc::flags::element_aligned);
            //     current_potential_result++;
            // }

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
            current_potential_result++;    // don't remove this

            // for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
            //     current_potential[i] += grad_0 * D[i];
            // }

            // iterator points to 10 after last loop

            // // TODO: only occurence of 10-19, don't cache
            // current_potential[10] += m_partner[0] * D[10];
            // current_potential[11] += m_partner[0] * D[11];
            // current_potential[12] += m_partner[0] * D[12];
            // current_potential[13] += m_partner[0] * D[13];
            // current_potential[14] += m_partner[0] * D[14];
            // current_potential[15] += m_partner[0] * D[15];
            // current_potential[16] += m_partner[0] * D[16];
            // current_potential[17] += m_partner[0] * D[17];
            // current_potential[18] += m_partner[0] * D[18];
            // current_potential[19] += m_partner[0] * D[19];

            tmp = current_potential_result.value() + m_partner[0] * D[10];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[11];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[12];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[13];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[14];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[15];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[16];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[17];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[18];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);
            current_potential_result++;
            tmp = current_potential_result.value() + m_partner[0] * D[19];
            Vc::where(mask, tmp).memstore(
                current_potential_result.pointer(), Vc::flags::element_aligned);

            // TODO: remove this when switching back to non-copy (reference-based)
            // approach

            // struct_of_array_taylor<expansion, real, 20> current_potential_result =
            //     potential_expansions_SoA.get_view(cell_flat_index_unpadded);

            ////////////// angular momentum correction, if enabled /////////////////////////

            if (type == RHO) {
                // struct_of_array_taylor<space_vector, real, 3> current_angular_correction_result =
                //     angular_corrections_SoA.get_view(cell_flat_index_unpadded);
                struct_of_array_iterator<space_vector, real, 3> current_angular_correction_result(
                    angular_corrections_SoA, cell_flat_index_unpadded);

                std::array<m2m_vector, NDIM> current_angular_correction;
                // for (size_t i = 0; i < current_angular_correction.size(); i++) {
                //     current_angular_correction[i] = 0.0;
                // }
                current_angular_correction[0] = 0.0;
                current_angular_correction[1] = 0.0;
                current_angular_correction[2] = 0.0;

                // for (integer a = 0; a != NDIM; ++a) {
                //     int const* abcd_idx_map = to_abcd_idx_map3[a];
                //     for (integer i = 0; i != 10; ++i) {
                //         const integer bcd_idx = bcd_idx_map[i];
                //         const m2m_vector tmp = D[abcd_idx_map[i]] * (factor[bcd_idx] * SIXTH);
                //         current_angular_correction[a] -= n0[bcd_idx] * tmp;
                //     }
                // }

                current_angular_correction[0] -= n0[10] * (D[20] * (factor[10] * SIXTH));
                current_angular_correction[0] -= n0[11] * (D[21] * (factor[11] * SIXTH));
                current_angular_correction[0] -= n0[12] * (D[22] * (factor[12] * SIXTH));
                current_angular_correction[0] -= n0[13] * (D[23] * (factor[13] * SIXTH));
                current_angular_correction[0] -= n0[14] * (D[24] * (factor[14] * SIXTH));
                current_angular_correction[0] -= n0[15] * (D[25] * (factor[15] * SIXTH));
                current_angular_correction[0] -= n0[16] * (D[26] * (factor[16] * SIXTH));
                current_angular_correction[0] -= n0[17] * (D[27] * (factor[17] * SIXTH));
                current_angular_correction[0] -= n0[18] * (D[28] * (factor[18] * SIXTH));
                current_angular_correction[0] -= n0[19] * (D[29] * (factor[19] * SIXTH));

                current_angular_correction[1] -= n0[10] * (D[21] * (factor[10] * SIXTH));
                current_angular_correction[1] -= n0[11] * (D[23] * (factor[11] * SIXTH));
                current_angular_correction[1] -= n0[12] * (D[24] * (factor[12] * SIXTH));
                current_angular_correction[1] -= n0[13] * (D[26] * (factor[13] * SIXTH));
                current_angular_correction[1] -= n0[14] * (D[27] * (factor[14] * SIXTH));
                current_angular_correction[1] -= n0[15] * (D[28] * (factor[15] * SIXTH));
                current_angular_correction[1] -= n0[16] * (D[30] * (factor[16] * SIXTH));
                current_angular_correction[1] -= n0[17] * (D[31] * (factor[17] * SIXTH));
                current_angular_correction[1] -= n0[18] * (D[32] * (factor[18] * SIXTH));
                current_angular_correction[1] -= n0[19] * (D[33] * (factor[19] * SIXTH));

                current_angular_correction[2] -= n0[10] * (D[22] * (factor[10] * SIXTH));
                current_angular_correction[2] -= n0[11] * (D[24] * (factor[11] * SIXTH));
                current_angular_correction[2] -= n0[12] * (D[25] * (factor[12] * SIXTH));
                current_angular_correction[2] -= n0[13] * (D[27] * (factor[13] * SIXTH));
                current_angular_correction[2] -= n0[14] * (D[28] * (factor[14] * SIXTH));
                current_angular_correction[2] -= n0[15] * (D[29] * (factor[15] * SIXTH));
                current_angular_correction[2] -= n0[16] * (D[31] * (factor[16] * SIXTH));
                current_angular_correction[2] -= n0[17] * (D[32] * (factor[17] * SIXTH));
                current_angular_correction[2] -= n0[18] * (D[33] * (factor[18] * SIXTH));
                current_angular_correction[2] -= n0[19] * (D[34] * (factor[19] * SIXTH));

                // for (size_t i = 0; i < current_angular_correction.size(); i++) {
                //     m2m_vector tmp =
                //         m2m_vector(*current_angular_correction_result,
                //         Vc::flags::element_aligned) +
                //         current_angular_correction[i];
                //     Vc::where(mask, tmp).memstore(
                //         *current_angular_correction_result, Vc::flags::element_aligned);
                //     current_angular_correction_result++;
                // }
                m2m_vector tmp =
                    current_angular_correction_result.value() + current_angular_correction[0];
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
    }
}    // namespace fmm
}    // namespace octotiger
