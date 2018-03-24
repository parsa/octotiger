#pragma once
#include "../common_kernel/interaction_constants.hpp"
#include "../cuda_util/cuda_global_def.hpp"
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {
        CUDA_CALLABLE_METHOD constexpr double factor[20] = {1.000000, 1.000000, 1.000000, 1.000000,
            1.000000, 2.000000, 2.000000, 1.000000, 2.000000, 1.000000, 1.000000, 3.000000,
            3.000000, 3.000000, 6.000000, 3.000000, 1.000000, 3.000000, 3.000000, 1.00000};
        CUDA_CALLABLE_METHOD constexpr double factor_half[20] = {factor[0] / 2.0, factor[1] / 2.0,
            factor[2] / 2.0, factor[3] / 2.0, factor[4] / 2.0, factor[5] / 2.0, factor[6] / 2.0,
            factor[7] / 2.0, factor[8] / 2.0, factor[9] / 2.0, factor[10] / 2.0, factor[11] / 2.0,
            factor[12] / 2.0, factor[13] / 2.0, factor[14] / 2.0, factor[15] / 2.0,
            factor[16] / 2.0, factor[17] / 2.0, factor[18] / 2.0, factor[19] / 2.0};
        CUDA_CALLABLE_METHOD constexpr double factor_sixth[20] = {factor[0] / 6.0, factor[1] / 6.0,
            factor[2] / 6.0, factor[3] / 6.0, factor[4] / 6.0, factor[5] / 6.0, factor[6] / 6.0,
            factor[7] / 6.0, factor[8] / 6.0, factor[9] / 6.0, factor[10] / 6.0, factor[11] / 6.0,
            factor[12] / 6.0, factor[13] / 6.0, factor[14] / 6.0, factor[15] / 6.0,
            factor[16] / 6.0, factor[17] / 6.0, factor[18] / 6.0, factor[19] / 6.0};

        template <typename T>
        CUDA_CALLABLE_METHOD constexpr inline T sqr(T const& val) noexcept {
            return val * val;
        }

        template <typename T, typename func>
        CUDA_CALLABLE_METHOD inline void compute_d_factors(T& d2, T& d3, T& X_00, T& X_11, T& X_22,
            T (&D_lower)[20], const T (&dX)[NDIM], func&& max) noexcept {
            X_00 = dX[0] * dX[0];
            X_11 = dX[1] * dX[1];
            X_22 = dX[2] * dX[2];

            T d0;
            T r2inv = T(1.0) / max(X_00 + X_11 + X_22, T(1.0e-20));

            d0 = -sqrt(r2inv);
            d2 = -3.0 * (-d0 * r2inv) * r2inv;
            d3 = -5.0 * d2 * r2inv;

            D_lower[0] = d0;
            D_lower[1] = dX[0] * (-d0 * r2inv);
            D_lower[2] = dX[1] * (-d0 * r2inv);
            D_lower[3] = dX[2] * (-d0 * r2inv);


            D_lower[4] = d2 * X_00;
            D_lower[4] += (-d0 * r2inv);
            D_lower[5] = d2 * dX[0] * dX[1];
            D_lower[6] = d2 * dX[0] * dX[2];

            D_lower[7] = d2 * X_11;
            D_lower[7] += (-d0 * r2inv);
            D_lower[8] = d2 * dX[1] * dX[2];

            D_lower[9] = d2 * X_22;
            D_lower[9] += (-d0 * r2inv);

            D_lower[10] = d3 * X_00 * dX[0];
            D_lower[10] += 3.0 * d2 * dX[0];
            D_lower[11] = d3 * X_00 * dX[1];
            D_lower[11] += d2 * dX[1];
            D_lower[12] = d3 * X_00 * dX[2];
            D_lower[12] += d2 * dX[2];

            D_lower[13] = d3 * dX[0] * X_11;
            D_lower[13] += d2 * dX[0];
            D_lower[14] = d3 * dX[0] * dX[1] * dX[2];

            D_lower[15] = d3 * dX[0] * X_22;
            D_lower[15] += d2 * dX[0];

            D_lower[16] = d3 * X_11 * dX[1];
            D_lower[16] += 3.0 * d2 * dX[1];

            D_lower[17] = d3 * X_11 * dX[2];
            D_lower[17] += d2 * dX[2];

            D_lower[18] = d3 * dX[1] * X_22;
            D_lower[18] += d2 * dX[1];

            D_lower[19] = d3 * X_22 * dX[2];
            D_lower[19] += 3.0 * d2 * dX[2];
        }
        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_interaction_multipole_non_rho0123(
            const T (&m_partner)[20], T (&tmpstore)[20], const T (&D_lower)[20]) noexcept {
            tmpstore[0] += m_partner[0] * D_lower[0];
            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[3] += m_partner[0] * D_lower[3];

            tmpstore[0] -= m_partner[1] * D_lower[1];
            tmpstore[1] -= m_partner[1] * D_lower[4];
            tmpstore[1] -= m_partner[1] * D_lower[5];
            tmpstore[1] -= m_partner[1] * D_lower[6];
            tmpstore[0] -= m_partner[2] * D_lower[2];
            tmpstore[2] -= m_partner[2] * D_lower[5];
            tmpstore[2] -= m_partner[2] * D_lower[7];
            tmpstore[2] -= m_partner[2] * D_lower[8];
            tmpstore[0] -= m_partner[3] * D_lower[3];
            tmpstore[3] -= m_partner[3] * D_lower[6];
            tmpstore[3] -= m_partner[3] * D_lower[8];
            tmpstore[3] -= m_partner[3] * D_lower[9];

            tmpstore[4] += m_partner[0] * D_lower[4];
            tmpstore[5] += m_partner[0] * D_lower[5];
            tmpstore[6] += m_partner[0] * D_lower[6];
            tmpstore[7] += m_partner[0] * D_lower[7];
            tmpstore[8] += m_partner[0] * D_lower[8];
            tmpstore[9] += m_partner[0] * D_lower[9];

            tmpstore[4] -= m_partner[1] * D_lower[10];
            tmpstore[5] -= m_partner[1] * D_lower[11];
            tmpstore[6] -= m_partner[1] * D_lower[12];
            tmpstore[7] -= m_partner[1] * D_lower[13];
            tmpstore[8] -= m_partner[1] * D_lower[14];
            tmpstore[9] -= m_partner[1] * D_lower[15];

            tmpstore[4] -= m_partner[2] * D_lower[11];
            tmpstore[5] -= m_partner[2] * D_lower[13];
            tmpstore[6] -= m_partner[2] * D_lower[14];
            tmpstore[7] -= m_partner[2] * D_lower[16];
            tmpstore[8] -= m_partner[2] * D_lower[17];
            tmpstore[9] -= m_partner[2] * D_lower[18];

            tmpstore[4] -= m_partner[3] * D_lower[12];
            tmpstore[5] -= m_partner[3] * D_lower[14];
            tmpstore[6] -= m_partner[3] * D_lower[15];
            tmpstore[7] -= m_partner[3] * D_lower[17];
            tmpstore[8] -= m_partner[3] * D_lower[18];
            tmpstore[9] -= m_partner[3] * D_lower[19];

            tmpstore[10] += m_partner[0] * D_lower[10];
            tmpstore[11] += m_partner[0] * D_lower[11];
            tmpstore[12] += m_partner[0] * D_lower[12];
            tmpstore[13] += m_partner[0] * D_lower[13];
            tmpstore[14] += m_partner[0] * D_lower[14];
            tmpstore[15] += m_partner[0] * D_lower[15];
            tmpstore[16] += m_partner[0] * D_lower[16];
            tmpstore[17] += m_partner[0] * D_lower[17];
            tmpstore[18] += m_partner[0] * D_lower[18];
            tmpstore[19] += m_partner[0] * D_lower[19];
        }

        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_interaction_multipole_non_rho456789(
            const T (&m_partner)[20], T (&tmpstore)[20], const T (&D_lower)[20]) noexcept {
            tmpstore[0] += m_partner[4] * (D_lower[4] * factor_half[4]);
            tmpstore[1] += m_partner[4] * (D_lower[10] * factor_half[4]);
            tmpstore[2] += m_partner[4] * (D_lower[11] * factor_half[4]);
            tmpstore[3] += m_partner[4] * (D_lower[12] * factor_half[4]);

            tmpstore[0] += m_partner[5] * (D_lower[5] * factor_half[5]);
            tmpstore[1] += m_partner[5] * (D_lower[11] * factor_half[5]);
            tmpstore[2] += m_partner[5] * (D_lower[13] * factor_half[5]);
            tmpstore[3] += m_partner[5] * (D_lower[14] * factor_half[5]);

            tmpstore[0] += m_partner[6] * (D_lower[6] * factor_half[6]);
            tmpstore[1] += m_partner[6] * (D_lower[12] * factor_half[6]);
            tmpstore[2] += m_partner[6] * (D_lower[14] * factor_half[6]);
            tmpstore[3] += m_partner[6] * (D_lower[15] * factor_half[6]);

            tmpstore[0] += m_partner[7] * (D_lower[7] * factor_half[7]);
            tmpstore[1] += m_partner[7] * (D_lower[13] * factor_half[7]);
            tmpstore[2] += m_partner[7] * (D_lower[16] * factor_half[7]);
            tmpstore[3] += m_partner[7] * (D_lower[17] * factor_half[7]);

            tmpstore[0] += m_partner[8] * (D_lower[8] * factor_half[8]);
            tmpstore[1] += m_partner[8] * (D_lower[14] * factor_half[8]);
            tmpstore[2] += m_partner[8] * (D_lower[17] * factor_half[8]);
            tmpstore[3] += m_partner[8] * (D_lower[18] * factor_half[8]);

            tmpstore[0] += m_partner[9] * (D_lower[9] * factor_half[9]);
            tmpstore[1] += m_partner[9] * (D_lower[15] * factor_half[9]);
            tmpstore[2] += m_partner[9] * (D_lower[18] * factor_half[9]);
            tmpstore[3] += m_partner[9] * (D_lower[19] * factor_half[9]);
        }
        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_interaction_multipole_non_rho_remainining(
            const T (&m_partner)[20], T (&tmpstore)[20], const T (&D_lower)[20]) noexcept {
            tmpstore[0] -= m_partner[10] * (D_lower[10] * factor_sixth[10]);
            tmpstore[0] -= m_partner[11] * (D_lower[11] * factor_sixth[11]);
            tmpstore[0] -= m_partner[12] * (D_lower[12] * factor_sixth[12]);
            tmpstore[0] -= m_partner[13] * (D_lower[13] * factor_sixth[13]);
            tmpstore[0] -= m_partner[14] * (D_lower[14] * factor_sixth[14]);
            tmpstore[0] -= m_partner[15] * (D_lower[15] * factor_sixth[15]);
            tmpstore[0] -= m_partner[16] * (D_lower[16] * factor_sixth[16]);
            tmpstore[0] -= m_partner[17] * (D_lower[17] * factor_sixth[17]);
            tmpstore[0] -= m_partner[18] * (D_lower[18] * factor_sixth[18]);
            tmpstore[0] -= m_partner[19] * (D_lower[19] * factor_sixth[19]);
        }
        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_interaction_sorted_non_rho(
            const T (&m_partner)[20], T (&tmpstore)[20], const T (&D_lower)[20]) noexcept {

            tmpstore[0] += m_partner[0] * D_lower[0];

            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[0] -= m_partner[1] * D_lower[1];

            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[0] -= m_partner[2] * D_lower[2];

            tmpstore[3] += m_partner[0] * D_lower[3];
            tmpstore[0] -= m_partner[3] * D_lower[3];

            tmpstore[1] -= m_partner[1] * D_lower[4];
            tmpstore[0] += m_partner[4] * (D_lower[4] * factor_half[4]);

            tmpstore[4] += m_partner[0] * D_lower[4];

            tmpstore[1] -= m_partner[1] * D_lower[5];
            tmpstore[2] -= m_partner[2] * D_lower[5];
            tmpstore[0] += m_partner[5] * (D_lower[5] * factor_half[5]);

            tmpstore[5] += m_partner[0] * D_lower[5];

            tmpstore[1] -= m_partner[1] * D_lower[6];
            tmpstore[3] -= m_partner[3] * D_lower[6];
            tmpstore[0] += m_partner[6] * (D_lower[6] * factor_half[6]);

            tmpstore[6] += m_partner[0] * D_lower[6];

            tmpstore[2] -= m_partner[2] * D_lower[7];
            tmpstore[0] += m_partner[7] * (D_lower[7] * factor_half[7]);

            tmpstore[7] += m_partner[0] * D_lower[7];


            tmpstore[2] -= m_partner[2] * D_lower[8];
            tmpstore[3] -= m_partner[3] * D_lower[8];
            tmpstore[0] += m_partner[8] * (D_lower[8] * factor_half[8]);

            tmpstore[8] += m_partner[0] * D_lower[8];


            tmpstore[3] -= m_partner[3] * D_lower[9];
            tmpstore[0] += m_partner[9] * (D_lower[9] * factor_half[9]);

            tmpstore[9] += m_partner[0] * D_lower[9];


          //logical cut


            tmpstore[1] += m_partner[4] * (D_lower[10] * factor_half[4]);
            tmpstore[0] -= m_partner[10] * (D_lower[10] * factor_sixth[10]);
            tmpstore[4] -= m_partner[1] * D_lower[10];
            tmpstore[10] += m_partner[0] * D_lower[10];


            tmpstore[2] += m_partner[4] * (D_lower[11] * factor_half[4]);
            tmpstore[1] += m_partner[5] * (D_lower[11] * factor_half[5]);
            tmpstore[0] -= m_partner[11] * (D_lower[11] * factor_sixth[11]);
            tmpstore[5] -= m_partner[1] * D_lower[11];
            tmpstore[4] -= m_partner[2] * D_lower[11];
            tmpstore[11] += m_partner[0] * D_lower[11];


            tmpstore[3] += m_partner[4] * (D_lower[12] * factor_half[4]);
            tmpstore[1] += m_partner[6] * (D_lower[12] * factor_half[6]);
            tmpstore[0] -= m_partner[12] * (D_lower[12] * factor_sixth[12]);
            tmpstore[6] -= m_partner[1] * D_lower[12];
            tmpstore[4] -= m_partner[3] * D_lower[12];
            tmpstore[12] += m_partner[0] * D_lower[12];


            tmpstore[2] += m_partner[5] * (D_lower[13] * factor_half[5]);
            tmpstore[1] += m_partner[7] * (D_lower[13] * factor_half[7]);
            tmpstore[0] -= m_partner[13] * (D_lower[13] * factor_sixth[13]);
            tmpstore[7] -= m_partner[1] * D_lower[13];
            tmpstore[5] -= m_partner[2] * D_lower[13];
            tmpstore[13] += m_partner[0] * D_lower[13];


            tmpstore[3] += m_partner[5] * (D_lower[14] * factor_half[5]);
            tmpstore[2] += m_partner[6] * (D_lower[14] * factor_half[6]);
            tmpstore[1] += m_partner[8] * (D_lower[14] * factor_half[8]);
            tmpstore[0] -= m_partner[14] * (D_lower[14] * factor_sixth[14]);
            tmpstore[8] -= m_partner[1] * D_lower[14];
            tmpstore[6] -= m_partner[2] * D_lower[14];
            tmpstore[5] -= m_partner[3] * D_lower[14];
            tmpstore[14] += m_partner[0] * D_lower[14];


            tmpstore[3] += m_partner[6] * (D_lower[15] * factor_half[6]);
            tmpstore[1] += m_partner[9] * (D_lower[15] * factor_half[9]);
            tmpstore[0] -= m_partner[15] * (D_lower[15] * factor_sixth[15]);
            tmpstore[9] -= m_partner[1] * D_lower[15];
            tmpstore[6] -= m_partner[3] * D_lower[15];
            tmpstore[15] += m_partner[0] * D_lower[15];


            tmpstore[2] += m_partner[7] * (D_lower[16] * factor_half[7]);
            tmpstore[0] -= m_partner[16] * (D_lower[16] * factor_sixth[16]);
            tmpstore[7] -= m_partner[2] * D_lower[16];
            tmpstore[16] += m_partner[0] * D_lower[16];


            tmpstore[3] += m_partner[7] * (D_lower[17] * factor_half[7]);
            tmpstore[2] += m_partner[8] * (D_lower[17] * factor_half[8]);
            tmpstore[0] -= m_partner[17] * (D_lower[17] * factor_sixth[17]);
            tmpstore[8] -= m_partner[2] * D_lower[17];
            tmpstore[7] -= m_partner[3] * D_lower[17];
            tmpstore[17] += m_partner[0] * D_lower[17];


            tmpstore[3] += m_partner[8] * (D_lower[18] * factor_half[8]);
            tmpstore[2] += m_partner[9] * (D_lower[18] * factor_half[9]);
            tmpstore[0] -= m_partner[18] * (D_lower[18] * factor_sixth[18]);
            tmpstore[9] -= m_partner[2] * D_lower[18];
            tmpstore[8] -= m_partner[3] * D_lower[18];
            tmpstore[18] += m_partner[0] * D_lower[18];


            tmpstore[3] += m_partner[9] * (D_lower[19] * factor_half[9]);
            tmpstore[0] -= m_partner[19] * (D_lower[19] * factor_sixth[19]);
            tmpstore[9] -= m_partner[3] * D_lower[19];
            tmpstore[19] += m_partner[0] * D_lower[19];



            //warning duplicate
            tmpstore[0] -= m_partner[10] * (D_lower[10] * factor_sixth[10]);
            tmpstore[0] -= m_partner[11] * (D_lower[11] * factor_sixth[11]);
            tmpstore[0] -= m_partner[12] * (D_lower[12] * factor_sixth[12]);
            tmpstore[0] -= m_partner[13] * (D_lower[13] * factor_sixth[13]);
            tmpstore[0] -= m_partner[14] * (D_lower[14] * factor_sixth[14]);
            tmpstore[0] -= m_partner[15] * (D_lower[15] * factor_sixth[15]);
            tmpstore[0] -= m_partner[16] * (D_lower[16] * factor_sixth[16]);
            tmpstore[0] -= m_partner[17] * (D_lower[17] * factor_sixth[17]);
            tmpstore[0] -= m_partner[18] * (D_lower[18] * factor_sixth[18]);
            tmpstore[0] -= m_partner[19] * (D_lower[19] * factor_sixth[19]);
        }

        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_interaction_multipole_non_rho(
            const T (&m_partner)[20], T (&tmpstore)[20], const T (&D_lower)[20]) noexcept {
            tmpstore[0] += m_partner[4] * (D_lower[4] * factor_half[4]);
            tmpstore[1] += m_partner[4] * (D_lower[10] * factor_half[4]);
            tmpstore[2] += m_partner[4] * (D_lower[11] * factor_half[4]);
            tmpstore[3] += m_partner[4] * (D_lower[12] * factor_half[4]);

            tmpstore[0] += m_partner[5] * (D_lower[5] * factor_half[5]);
            tmpstore[1] += m_partner[5] * (D_lower[11] * factor_half[5]);
            tmpstore[2] += m_partner[5] * (D_lower[13] * factor_half[5]);
            tmpstore[3] += m_partner[5] * (D_lower[14] * factor_half[5]);

            tmpstore[0] += m_partner[6] * (D_lower[6] * factor_half[6]);
            tmpstore[1] += m_partner[6] * (D_lower[12] * factor_half[6]);
            tmpstore[2] += m_partner[6] * (D_lower[14] * factor_half[6]);
            tmpstore[3] += m_partner[6] * (D_lower[15] * factor_half[6]);

            tmpstore[0] += m_partner[7] * (D_lower[7] * factor_half[7]);
            tmpstore[1] += m_partner[7] * (D_lower[13] * factor_half[7]);
            tmpstore[2] += m_partner[7] * (D_lower[16] * factor_half[7]);
            tmpstore[3] += m_partner[7] * (D_lower[17] * factor_half[7]);

            tmpstore[0] += m_partner[8] * (D_lower[8] * factor_half[8]);
            tmpstore[1] += m_partner[8] * (D_lower[14] * factor_half[8]);
            tmpstore[2] += m_partner[8] * (D_lower[17] * factor_half[8]);
            tmpstore[3] += m_partner[8] * (D_lower[18] * factor_half[8]);

            tmpstore[0] += m_partner[9] * (D_lower[9] * factor_half[9]);
            tmpstore[1] += m_partner[9] * (D_lower[15] * factor_half[9]);
            tmpstore[2] += m_partner[9] * (D_lower[18] * factor_half[9]);
            tmpstore[3] += m_partner[9] * (D_lower[19] * factor_half[9]);

            tmpstore[0] -= m_partner[10] * (D_lower[10] * factor_sixth[10]);
            tmpstore[0] -= m_partner[11] * (D_lower[11] * factor_sixth[11]);
            tmpstore[0] -= m_partner[12] * (D_lower[12] * factor_sixth[12]);
            tmpstore[0] -= m_partner[13] * (D_lower[13] * factor_sixth[13]);
            tmpstore[0] -= m_partner[14] * (D_lower[14] * factor_sixth[14]);
            tmpstore[0] -= m_partner[15] * (D_lower[15] * factor_sixth[15]);
            tmpstore[0] -= m_partner[16] * (D_lower[16] * factor_sixth[16]);
            tmpstore[0] -= m_partner[17] * (D_lower[17] * factor_sixth[17]);
            tmpstore[0] -= m_partner[18] * (D_lower[18] * factor_sixth[18]);
            tmpstore[0] -= m_partner[19] * (D_lower[19] * factor_sixth[19]);

            tmpstore[4] += m_partner[0] * D_lower[4];
            tmpstore[5] += m_partner[0] * D_lower[5];
            tmpstore[6] += m_partner[0] * D_lower[6];
            tmpstore[7] += m_partner[0] * D_lower[7];
            tmpstore[8] += m_partner[0] * D_lower[8];
            tmpstore[9] += m_partner[0] * D_lower[9];

            tmpstore[4] -= m_partner[1] * D_lower[10];
            tmpstore[5] -= m_partner[1] * D_lower[11];
            tmpstore[6] -= m_partner[1] * D_lower[12];
            tmpstore[7] -= m_partner[1] * D_lower[13];
            tmpstore[8] -= m_partner[1] * D_lower[14];
            tmpstore[9] -= m_partner[1] * D_lower[15];

            tmpstore[4] -= m_partner[2] * D_lower[11];
            tmpstore[5] -= m_partner[2] * D_lower[13];
            tmpstore[6] -= m_partner[2] * D_lower[14];
            tmpstore[7] -= m_partner[2] * D_lower[16];
            tmpstore[8] -= m_partner[2] * D_lower[17];
            tmpstore[9] -= m_partner[2] * D_lower[18];

            tmpstore[4] -= m_partner[3] * D_lower[12];
            tmpstore[5] -= m_partner[3] * D_lower[14];
            tmpstore[6] -= m_partner[3] * D_lower[15];
            tmpstore[7] -= m_partner[3] * D_lower[17];
            tmpstore[8] -= m_partner[3] * D_lower[18];
            tmpstore[9] -= m_partner[3] * D_lower[19];
            tmpstore[10] += m_partner[0] * D_lower[10];
            tmpstore[11] += m_partner[0] * D_lower[11];
            tmpstore[12] += m_partner[0] * D_lower[12];
            tmpstore[13] += m_partner[0] * D_lower[13];
            tmpstore[14] += m_partner[0] * D_lower[14];
            tmpstore[15] += m_partner[0] * D_lower[15];
            tmpstore[16] += m_partner[0] * D_lower[16];
            tmpstore[17] += m_partner[0] * D_lower[17];
            tmpstore[18] += m_partner[0] * D_lower[18];
            tmpstore[19] += m_partner[0] * D_lower[19];
        }

        template <typename T>
        CUDA_CALLABLE_METHOD inline void compute_interaction_multipole_rho(const T& d2, const T& d3,
            const T& X_00, const T& X_11, const T& X_22, const T (&m_partner)[20],
            const T (&m_cell)[11], const T (&dX)[NDIM], T (&tmp_corrections)[NDIM]) noexcept {
            T n0_constant = m_partner[0] / m_cell[0];

            T D_upper[6];

            D_upper[0] = dX[0] * dX[0] * d3 + 2.0 * d2;
            D_upper[0] += d2;
            D_upper[0] += 5.0 * d3 * X_00;
            D_upper[1] = 3.0 * d3 * dX[0] * dX[1];
            D_upper[2] = 3.0 * d3 * dX[0] * dX[2];
            T n0_tmp = m_partner[10] - m_cell[1] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[0] * factor_sixth[10]);
            tmp_corrections[1] -= n0_tmp * (D_upper[1] * factor_sixth[10]);
            tmp_corrections[2] -= n0_tmp * (D_upper[2] * factor_sixth[10]);

            D_upper[0] = d2;
            D_upper[0] += d3 * X_11;
            D_upper[0] += d3 * X_00;
            D_upper[3] = d3 * dX[1] * dX[2];

            n0_tmp = m_partner[11] - m_cell[2] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[1] * factor_sixth[11]);
            tmp_corrections[1] -= n0_tmp * (D_upper[0] * factor_sixth[11]);
            tmp_corrections[2] -= n0_tmp * (D_upper[3] * factor_sixth[11]);

            D_upper[1] = d2;
            D_upper[1] += d3 * X_22;
            D_upper[1] += d3 * X_00;

            n0_tmp = m_partner[12] - m_cell[3] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[2] * factor_sixth[12]);
            tmp_corrections[1] -= n0_tmp * (D_upper[3] * factor_sixth[12]);
            tmp_corrections[2] -= n0_tmp * (D_upper[1] * factor_sixth[12]);

            D_upper[2] = 3.0 * d3 * dX[0] * dX[1];
            D_upper[4] = d3 * dX[0] * dX[2];

            n0_tmp = m_partner[13] - m_cell[4] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[0] * factor_sixth[13]);
            tmp_corrections[1] -= n0_tmp * (D_upper[2] * factor_sixth[13]);
            tmp_corrections[2] -= n0_tmp * (D_upper[4] * factor_sixth[13]);

            D_upper[0] = d3 * dX[0] * dX[1];

            n0_tmp = m_partner[14] - m_cell[5] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[3] * factor_sixth[14]);
            tmp_corrections[1] -= n0_tmp * (D_upper[4] * factor_sixth[14]);
            tmp_corrections[2] -= n0_tmp * (D_upper[0] * factor_sixth[14]);

            D_upper[3] = 3.0 * d3 * dX[0] * dX[2];

            n0_tmp = m_partner[15] - m_cell[6] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[1] * factor_sixth[15]);
            tmp_corrections[1] -= n0_tmp * (D_upper[0] * factor_sixth[15]);
            tmp_corrections[2] -= n0_tmp * (D_upper[3] * factor_sixth[15]);

            D_upper[1] = dX[1] * dX[1] * d3 + 2.0 * d2;
            D_upper[1] += d2;
            D_upper[1] += 5.0 * d3 * X_11;

            D_upper[5] = 3.0 * d3 * dX[1] * dX[2];

            n0_tmp = m_partner[16] - m_cell[7] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[2] * factor_sixth[16]);
            tmp_corrections[1] -= n0_tmp * (D_upper[1] * factor_sixth[16]);
            tmp_corrections[2] -= n0_tmp * (D_upper[5] * factor_sixth[16]);

            D_upper[2] = d2;
            D_upper[2] += d3 * X_22;
            D_upper[2] += d3 * X_11;

            n0_tmp = m_partner[17] - m_cell[8] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[4] * factor_sixth[17]);
            tmp_corrections[1] -= n0_tmp * (D_upper[5] * factor_sixth[17]);
            tmp_corrections[2] -= n0_tmp * (D_upper[2] * factor_sixth[17]);

            D_upper[4] = 3.0 * d3 * dX[1] * dX[2];

            n0_tmp = m_partner[18] - m_cell[9] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[0] * factor_sixth[18]);
            tmp_corrections[1] -= n0_tmp * (D_upper[2] * factor_sixth[18]);
            tmp_corrections[2] -= n0_tmp * (D_upper[4] * factor_sixth[18]);

            D_upper[0] = dX[2] * dX[2] * d3 + 2.0 * d2;
            D_upper[0] += d2;
            D_upper[0] += 5.0 * d3 * X_22;

            n0_tmp = m_partner[19] - m_cell[10] * n0_constant;

            tmp_corrections[0] -= n0_tmp * (D_upper[3] * factor_sixth[19]);
            tmp_corrections[1] -= n0_tmp * (D_upper[4] * factor_sixth[19]);
            tmp_corrections[2] -= n0_tmp * (D_upper[0] * factor_sixth[19]);
        }

        template <typename T, typename func>
        CUDA_CALLABLE_METHOD inline void compute_kernel_rho(T (&X)[NDIM], T (&Y)[NDIM],
            T (&m_partner)[20], T (&tmpstore)[20], T (&tmp_corrections)[3], T (&m_cell)[11],
            func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;
            T D_lower[20];

            compute_d_factors(d2, d3, X_00, X_11, X_22, D_lower, dX, max);

            tmpstore[0] += m_partner[0] * D_lower[0];
            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[3] += m_partner[0] * D_lower[3];
            compute_interaction_multipole_non_rho(m_partner, tmpstore, D_lower);

            compute_interaction_multipole_rho(
                d2, d3, X_00, X_11, X_22, m_partner, m_cell, dX, tmp_corrections);
        }

        template <typename T, typename func>
        CUDA_CALLABLE_METHOD inline void compute_kernel_angular_corrections(T (&X)[NDIM], T (&Y)[NDIM],
            T (&m_partner)[20],  T (&tmp_corrections)[3], T (&m_cell)[11],
            func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;

            X_00 = dX[0] * dX[0];
            X_11 = dX[1] * dX[1];
            X_22 = dX[2] * dX[2];

            T r2 = X_00 + X_11 + X_22;
            T r2inv = T(1.0) / max(r2, T(1.0e-20));

            d2 = -3.0 * sqrt(r2inv) * r2inv * r2inv;
            d3 = -5.0 * sqrt(r2inv) * r2inv * r2inv;

            compute_interaction_multipole_rho(
                d2, d3, X_00, X_11, X_22, m_partner, m_cell, dX, tmp_corrections);
        }

        template <typename T, typename func>
        CUDA_CALLABLE_METHOD inline void compute_kernel_non_rho(T (&X)[NDIM], T (&Y)[NDIM],
            T (&m_partner)[20], T (&tmpstore)[20], func&& max) noexcept {
            T dX[NDIM];
            dX[0] = X[0] - Y[0];
            dX[1] = X[1] - Y[1];
            dX[2] = X[2] - Y[2];

            T X_00, X_11, X_22;
            T d2, d3;
            T D_lower[20];
            compute_d_factors(d2, d3, X_00, X_11, X_22, D_lower, dX, max);

            tmpstore[0] += m_partner[0] * D_lower[0];
            tmpstore[1] += m_partner[0] * D_lower[1];
            tmpstore[2] += m_partner[0] * D_lower[2];
            tmpstore[3] += m_partner[0] * D_lower[3];
            tmpstore[0] -= m_partner[1] * D_lower[1];
            tmpstore[1] -= m_partner[1] * D_lower[4];
            tmpstore[1] -= m_partner[1] * D_lower[5];
            tmpstore[1] -= m_partner[1] * D_lower[6];
            tmpstore[0] -= m_partner[2] * D_lower[2];
            tmpstore[2] -= m_partner[2] * D_lower[5];
            tmpstore[2] -= m_partner[2] * D_lower[7];
            tmpstore[2] -= m_partner[2] * D_lower[8];
            tmpstore[0] -= m_partner[3] * D_lower[3];
            tmpstore[3] -= m_partner[3] * D_lower[6];
            tmpstore[3] -= m_partner[3] * D_lower[8];
            tmpstore[3] -= m_partner[3] * D_lower[9];
            compute_interaction_multipole_non_rho(m_partner, tmpstore, D_lower);
        }
    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
