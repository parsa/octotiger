#pragma once

#include "geometry.hpp"
#include "interaction_constants.hpp"
#include "multiindex.hpp"

namespace octotiger {
namespace fmm {

    // stride for multiple outer cells (and/or padding)
    inline size_t to_inner_flat_index_padded(const multiindex& m) {
        return m.x * PADDED_STRIDE * PADDED_STRIDE + m.y * PADDED_STRIDE + m.z;
    }

    // strides are only valid for single cell! (no padding)
    inline size_t to_inner_flat_index_not_padded(const multiindex& m) {
        return m.x * INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION +
            m.y * INNER_CELLS_PER_DIRECTION + m.z;
    }

    // meant to iterate the input data structure
    template <typename F>
    void iterate_inner_cells_padded(F f) {
        for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
            for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                    const multiindex m(i0 + INNER_CELLS_PADDING_DEPTH,
                        i1 + INNER_CELLS_PADDING_DEPTH, i2 + INNER_CELLS_PADDING_DEPTH);
                    const size_t inner_flat_index = to_inner_flat_index_padded(m);
                    const multiindex m_unpadded(i0, i1, i2);
                    const size_t inner_flat_index_unpadded =
                        to_inner_flat_index_not_padded(m_unpadded);
                    f(m, inner_flat_index, m_unpadded, inner_flat_index_unpadded);
                }
            }
        }
    }

    // meant to iterate the input data structure
    template <typename F>
    void iterate_inner_cells_padding(const geo::direction& dir, F f) {
        // TODO: implementation not finished
        for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
            for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                    const multiindex m(
                        i0 + INNER_CELLS_PADDING_DEPTH + dir[0] * INNER_CELLS_PADDING_DEPTH,
                        i1 + INNER_CELLS_PADDING_DEPTH + dir[1] * INNER_CELLS_PADDING_DEPTH,
                        i2 + INNER_CELLS_PADDING_DEPTH + dir[2] * INNER_CELLS_PADDING_DEPTH);
                    const size_t inner_flat_index = to_inner_flat_index_padded(m);
                    const multiindex m_unpadded(i0, i1, i2);
                    const size_t inner_flat_index_unpadded =
                        to_inner_flat_index_not_padded(m_unpadded);
                    f(m, inner_flat_index, m_unpadded, inner_flat_index_unpadded);
                }
            }
        }
    }

    // meant to iterate the output data structure
    template <typename F>
    void iterate_inner_cells_not_padded(F f) {
        for (size_t i0 = 0; i0 < INNER_CELLS_PER_DIRECTION; i0++) {
            for (size_t i1 = 0; i1 < INNER_CELLS_PER_DIRECTION; i1++) {
                for (size_t i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; i2++) {
                    multiindex i(i0, i1, i2);
                    size_t inner_flat_index = to_inner_flat_index_not_padded(i);
                    f(i, inner_flat_index);
                }
            }
        }
    }

}    // namespace fmm
}    // namespace octotiger
