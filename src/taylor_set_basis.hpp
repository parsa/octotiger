// include by talor.hpp

template <>
inline void taylor<5, simd_vector>::set_basis(const std::array<simd_vector, NDIM>& X) {
    constexpr integer N = 5;
    using T = simd_vector;
    // PROF_BEGIN;

    // also highly optimized

    // A is D in the paper in formula (6)
    taylor<N, T>& A = *this;

    const T r2 = sqr(X[0]) + sqr(X[1]) + sqr(X[2]);
    T r2inv = 0.0;
    for (volatile integer i = 0; i != simd_len; ++i) {
        if (r2[i] > 0.0) {
            r2inv[i] = ONE / std::max(r2[i], 1.0e-20);
        }
    }

    // parts of formula (6)
    const T d0 = -sqrt(r2inv);
    // parts of formula (7)
    const T d1 = -d0 * r2inv;
    // parts of formula (8)
    const T d2 = T(-3) * d1 * r2inv;
    // parts of  formula (9)
    const T d3 = T(-5) * d2 * r2inv;
    //     const T d4 = -T(7) * d3 * r2inv;

    // formula (6)
    A[0] = d0;

    // formula (7)
    // for (integer i = taylor_sizes[0], a = 0; a != NDIM; ++a, ++i) {
    //     A[i] = X[a] * d1;
    // }
    
    A[1] = X[0] * d1;
    A[2] = X[1] * d1;
    A[3] = X[2] * d1;
    
    // formula (8)
    // for (integer i = taylor_sizes[1], a = 0; a != NDIM; ++a) { //i = 1, a = 0
    //     T const Xad2 = X[a] * d2;
    //     for (integer b = a; b != NDIM; ++b, ++i) { // b = 0
    //         A[i] = Xad2 * X[b];
    //     }
    // }

    A[4] = X[0] * d2 * X[0]; 
    A[5] = X[0] * d2 * X[1]; 
    A[6] = X[0] * d2 * X[2]; 
    
    A[7] = X[1] * d2 * X[1];
    A[8] = X[1] * d2 * X[2];
    
    A[9] = X[2] * d2 * X[2];

    
    // // formula (9)
    // for (integer i = taylor_sizes[2], a = 0; a != NDIM; ++a) {
    //     T const Xad3 = X[a] * d3;
    //     for (integer b = a; b != NDIM; ++b) {
    //         T const Xabd3 = Xad3 * X[b];
    //         for (integer c = b; c != NDIM; ++c, ++i) {
    //             A[i] = Xabd3 * X[c];
    //         }
    //     }
    // }

    A[10] = d3 * X[0] * X[0] * X[0]; // a == 0
    A[11] = d3 * X[0] * X[0] * X[1];
    A[12] = d3 * X[0] * X[0] * X[2];
    
    A[13] = d3 * X[0] * X[1] * X[1];
    A[14] = d3 * X[0] * X[1] * X[2];
    
    A[15] = d3 * X[0] * X[2] * X[2];
 
    A[16] = d3 * X[1] * X[1] * X[1]; // a == 1
    A[17] = d3 * X[1] * X[1] * X[2];
    
    A[18] = d3 * X[1] * X[2] * X[2];
    
    A[19] = d3 * X[2] * X[2] * X[2]; // a == 2

    // formula (19)

    // // set the coefficients to zero that are calculated next
    // for (integer i = taylor_sizes[3]; i != taylor_sizes[4]; ++i) {
    //     A[i] = ZERO;
    // }

    auto const d22 = 2.0 * d2;

    // for (integer i = taylor_sizes[0]; i != taylor_sizes[1]; ++i) {
    //     A[to_aa[i]] += d1;
    //     integer const to_a_idx = to_a[i];
    //     A[to_aaa[i]] += X[to_a_idx] * d2;
    //     A[to_aaaa[i]] += sqr(X[to_a_idx]) * d3 + d22;
    // }


    A[4] += d1;
    A[7] += d1;
    A[9] += d1;
    
    A[10] += X[0] * d2;
    A[16] += X[1] * d2;
    A[19] += X[2] * d2;

    // A[to_aaaa[i]] += sqr(X[to_a[i]]) * d3 + d22;
    A[20] += sqr(X[0]) * d3 + d22;
    A[30] += sqr(X[1]) * d3 + d22;
    A[34] += sqr(X[2]) * d3 + d22;
    
    
    // for (integer i = taylor_sizes[1]; i != taylor_sizes[2]; ++i) {
    //     integer const to_a_idx = to_a[i];
    //     integer const to_b_idx = to_b[i];
    //     auto const Xabd3 = X[to_a_idx] * X[to_b_idx] * d3;
    //     // A[to_aab[i]] += X[to_b_idx] * d2;
    //     A[to_abb[i]] += X[to_a_idx] * d2;
    //     A[to_aaab[i]] += Xabd3;
    //     A[to_abbb[i]] += Xabd3;
    //     A[to_aabb[i]] += d2;
    // }
    
    A[10] += X[0] * d2;
    A[11] += X[1] * d2;
    A[12] += X[2] * d2;
    A[16] += X[1] * d2;
    A[17] += X[2] * d2;
    A[19] += X[2] * d2;
    
    A[10] += X[0] * d2;
    A[13] += X[0] * d2;
    A[15] += X[0] * d2;
    A[16] += X[1] * d2;
    A[18] += X[1] * d2;
    A[19] += X[2] * d2;
    
    A[20] += X[0] * X[0] * d3;
    A[21] += X[0] * X[1] * d3;
    A[22] += X[0] * X[2] * d3;
    A[30] += X[1] * X[1] * d3;
    A[31] += X[1] * X[2] * d3;
    A[34] += X[2] * X[2] * d3;
    
    A[20] += X[0] * X[0] * d3;
    A[26] += X[0] * X[1] * d3;
    A[29] += X[0] * X[2] * d3;
    A[30] += X[1] * X[1] * d3;
    A[33] += X[1] * X[2] * d3;
    A[34] += X[2] * X[2] * d3;
   
    A[20] += d2;
    A[23] += d2;
    A[25] += d2;
    A[30] += d2;
    A[32] += d2;
    A[34] += d2;
        
    // for (integer i = taylor_sizes[2]; i != taylor_sizes[3]; ++i) {
    //     integer const to_a_idx = to_a[i];
    //     integer const to_c_idx = to_c[i];
    //     auto const Xbd3 = X[to_b[i]] * d3;
    //     A[to_aabc[i]] += Xbd3 * X[to_c_idx];
    //     A[to_abbc[i]] += X[to_a_idx] * X[to_c_idx] * d3;
    //     A[to_abcc[i]] += X[to_a_idx] * Xbd3;
    // }

    A[20] += d3 * X[0] * X[0];
    A[21] += d3 * X[0] * X[1];
    A[22] += d3 * X[0] * X[2];
    A[23] += d3 * X[1] * X[1];
    A[24] += d3 * X[1] * X[2];
    A[25] += d3 * X[2] * X[2];
    A[30] += d3 * X[1] * X[1];
    A[31] += d3 * X[1] * X[2];
    A[32] += d3 * X[2] * X[2];
    A[34] += d3 * X[2] * X[2];
    
    A[20] += X[0] * X[0] * d3;
    A[21] += X[0] * X[1] * d3;
    A[22] += X[0] * X[2] * d3;
    A[26] += X[0] * X[1] * d3;
    A[27] += X[0] * X[2] * d3;
    A[29] += X[0] * X[2] * d3;
    A[30] += X[1] * X[1] * d3;
    A[31] += X[1] * X[2] * d3;
    A[33] += X[1] * X[2] * d3;
    A[34] += X[2] * X[2] * d3;
    
    A[20] += X[0] * X[0] * d3;
    A[23] += X[0] * X[0] * d3;
    A[25] += X[0] * X[0] * d3;
    A[26] += X[0] * X[1] * d3;
    A[28] += X[0] * X[1] * d3;
    A[29] += X[0] * X[2] * d3;
    A[30] += X[1] * X[1] * d3;
    A[32] += X[1] * X[1] * d3;
    A[33] += X[1] * X[2] * d3;
    A[34] += X[2] * X[2] * d3;
       
    // PROF_END;
}




//     for (integer a = 0; a != NDIM; a++) {
//         auto const Xad2 = X[a] * d2;
//         auto const Xad3 = X[a] * d3;
//         A(a, a) += d1;
//         A(a, a, a) += Xad2;
//         A(a, a, a, a) += Xad3 * X[a] + d22;
//         for (integer b = a; b != NDIM; b++) {
//             auto const Xabd3 = Xad3 * X[b];
//             auto const Xbd3 = X[b] * d3;
//             A(a, a, b) += X[b] * d2;
//             A(a, b, b) += Xad2;
//             A(a, a, a, b) += Xabd3;
//             A(a, b, b, b) += Xabd3;
//             A(a, a, b, b) += d2;
//             for (integer c = b; c != NDIM; c++) {
//                 A(a, a, b, c) += Xbd3 * X[c];
//                 A(a, b, b, c) += Xad3 * X[c];
//                 A(a, b, c, c) += Xabd3;
//             }
//         }
//     }
