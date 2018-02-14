#pragma once

//#include "defs.hpp"
#include "interaction_constants.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#ifndef NVCC
#define __device__
#endif

namespace octotiger {
namespace fmm {

    // is template to allow for vectorization
    template <typename T = int32_t>
    class multiindex
    {
    public:
        T x;
        T y;
        T z;

        __device__ multiindex(T x, T y, T z)
          : x(x)
          , y(y)
          , z(z) {}

        template <typename U>
        __device__ multiindex(const multiindex<U>& other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        __device__ inline double length() const {
            return sqrt(static_cast<double>(x * x + y * y + z * z));
        }

        __device__ inline bool compare(multiindex& other) {
            if (this->x == other.x && this->y == other.y && this->z == other.z) {
                return true;
            } else {
                return false;
            }
        }

        // set this multiindex to the next coarser level index
        // template <typename U>
        __device__ void transform_coarse() {
            const T patch_size = static_cast<typename T::value_type>(INX);
            const T subtract = static_cast<typename T::value_type>(INX / 2);

            x = ((x + patch_size) >> 1) - subtract;
            y = ((y + patch_size) >> 1) - subtract;
            z = ((z + patch_size) >> 1) - subtract;
        }

        // template <>
    };

    __device__ inline multiindex<> flat_index_to_multiindex_not_padded(size_t flat_index) {
        size_t x = flat_index / (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        flat_index %= (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        size_t y = flat_index / INNER_CELLS_PER_DIRECTION;
        flat_index %= INNER_CELLS_PER_DIRECTION;
        size_t z = flat_index;
        multiindex<> m(x, y, z);
        return m;
    }

    __device__ inline multiindex<> flat_index_to_multiindex_padded(size_t flat_index) {
        size_t x = flat_index / (PADDED_STRIDE * PADDED_STRIDE);
        flat_index %= (PADDED_STRIDE * PADDED_STRIDE);
        size_t y = flat_index / PADDED_STRIDE;
        flat_index %= PADDED_STRIDE;
        size_t z = flat_index;
        multiindex<> m(x, y, z);
        return m;
    }

    /** stride for multiple outer cells (and/or padding)
     * Note: for m2m_int_vector and integer
     * Note: returns uint32_t vector because of Vc limitation */
    template <typename T>
    __device__ inline T to_flat_index_padded(const multiindex<T>& m) {
        return m.x * PADDED_STRIDE * PADDED_STRIDE + m.y * PADDED_STRIDE + m.z;
    }

    /** are only valid for single cell! (no padding)
     * Note: for m2m_int_vector and integer
     * Note: returns uint32_t vector because of Vc limitation */
    template <typename T>
    __device__ inline T to_inner_flat_index_not_padded(const multiindex<T>& m) {
        return m.x * INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION +
            m.y * INNER_CELLS_PER_DIRECTION + m.z;
    }

    struct two_phase_stencil
    {
        std::vector<multiindex<>> stencil_elements;
        std::vector<bool> stencil_phase_indicator;
    };
}    // namespace fmm
}    // namespace octotiger

template <typename T>
std::ostream& operator<<(std::ostream& os, const octotiger::fmm::multiindex<T>& m) {
    return os << m.x << ", " << m.y << ", " << m.z;
}
