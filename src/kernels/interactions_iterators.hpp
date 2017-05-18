#pragma once

#include "geometry.hpp"
#include "interaction_constants.hpp"
#include "multiindex.hpp"

namespace octotiger {
namespace fmm {

    inline multiindex inner_flat_index_to_multiindex(size_t flat_index) {
        size_t x = flat_index / (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        flat_index %= (INNER_CELLS_PER_DIRECTION * INNER_CELLS_PER_DIRECTION);
        size_t y = flat_index / INNER_CELLS_PER_DIRECTION;
        flat_index %= INNER_CELLS_PER_DIRECTION;
        size_t z = flat_index;
        multiindex m(x, y, z);
        return m;
    }

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
    void iterate_inner_cells_padded(const F& f) {
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
    void iterate_inner_cells_padding(const geo::direction& dir, const F& f) {
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
    void iterate_inner_cells_not_padded(const F& f) {
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

    template <typename component_printer>
    void print_layered_not_padded(bool print_index, const component_printer& printer) {
        iterate_inner_cells_not_padded([&printer, print_index](multiindex& i, size_t flat_index) {
            if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
            }
            // std::cout << this->potential_expansions[flat_index];
            if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
                std::cout << ", ";
            }
            if (print_index) {
                std::cout << " (" << i << ") = ";
            }
            printer(i, flat_index);
            if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                std::cout << std::endl;
            }
        });
    }

    template <typename component_printer>
    void print_layered_padded(bool print_index, const component_printer& printer) {
        iterate_inner_cells_padded(
            [&printer, print_index](const multiindex& i, const size_t flat_index,
                const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
                }
                // std::cout << this->potential_expansions[flat_index];
                if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
                    std::cout << ", ";
                }
                if (print_index) {
                    std::cout << " (" << i << ") = ";
                }
                printer(i, flat_index, i_unpadded, flat_index_unpadded);
                if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << std::endl;
                }
            });
    }

    template <typename component_printer>
    void print_layered_padding(
        geo::direction& dir, bool print_index, const component_printer& printer) {
        iterate_inner_cells_padding(
            dir, [&printer, print_index](const multiindex& i, const size_t flat_index,
                     const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                if (i.y % INNER_CELLS_PER_DIRECTION == 0 && i.z % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << "-------- next layer: " << i.x << "---------" << std::endl;
                }
                // std::cout << this->potential_expansions[flat_index];
                if (i.z % INNER_CELLS_PER_DIRECTION != 0) {
                    std::cout << ", ";
                }
                if (print_index) {
                    std::cout << " (" << i << ") = ";
                }
                printer(i, flat_index, i_unpadded, flat_index_unpadded);
                if ((i.z + 1) % INNER_CELLS_PER_DIRECTION == 0) {
                    std::cout << std::endl;
                }
            });
    }

    template <typename compare_functional>
    bool compare_padded_with_non_padded(const compare_functional& c) {
        bool all_ok = true;
        iterate_inner_cells_padded([&c, &all_ok](const multiindex& i, const size_t flat_index,
            const multiindex& i_unpadded, const size_t flat_index_unpadded) {
            bool ok = c(i, flat_index, i_unpadded, flat_index_unpadded);
            if (all_ok) {
                all_ok = ok;
            }
        });
        return all_ok;
    }

    template <typename compare_functional>
    bool compare_non_padded_with_non_padded(const compare_functional& c) {
        bool all_ok = true;
        iterate_inner_cells_not_padded(
            [&c, &all_ok](const multiindex& i_unpadded, const size_t flat_index_unpadded) {
                bool ok = c(i_unpadded, flat_index_unpadded);
                if (all_ok) {
                    all_ok = ok;
                }
            });
        return all_ok;
    }

}    // namespace fmm
}    // namespace octotiger
