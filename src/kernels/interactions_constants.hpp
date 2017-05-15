#pragma once

namespace octotiger {

namespace detail {

    constexpr uint64_t const_pow(uint64_t base, uint64_t exp) {
        if (exp >= 1) {
            return base * const_pow(base, exp - 1);
        } else {
            return 1;
        }
    }
}

namespace fmm {

    // constants from defs.hpp in deobfuscated form
    constexpr uint64_t DIMENSION = 3;

    // number of expansions in each cell per direction
    constexpr uint64_t INNER_CELLS_PER_DIRECTION = 8;

    // number of expansions in each cell (inner cells)
    constexpr uint64_t INNER_CELLS = detail::const_pow(INNER_CELLS_PER_DIRECTION, DIMENSION);

    // TODO: change this to 4 as soon as simplified padding works
    constexpr uint64_t INNER_CELLS_PADDING_DEPTH = 8;

    // constexpr uint64_t PADDING_EDGE_SURFACE_SINGLE_DIRECTION =
    //     detail::const_pow(INNER_CELLS_PER_DIRECTION, DIMENSION - 1);
    constexpr uint64_t PADDED_STRIDE = INNER_CELLS_PER_DIRECTION + 2 * INNER_CELLS_PADDING_DEPTH;

    // total number of cells in the padding layer: boundary surfaces plus vertices
    // constexpr uint64_t VERTICES = detail::const_pow(2, DIMENSION);
    // constexpr uint64_t EXPANSION_EDGE_COUNT =
    //     2 * DIMENSION * PADDING_EDGE_SURFACE_SINGLE_DIRECTION * INNER_CELLS_PADDING_DEPTH;
    // constexpr uint64_t EXPANSION_VERTICES_COUNT =
    //     detail::const_pow(2, DIMENSION) * detail::const_pow(INNER_CELLS_PADDING_DEPTH,
    //     DIMENSION);
    constexpr uint64_t EXPANSION_COUNT_PADDED = detail::const_pow(PADDED_STRIDE, DIMENSION);
    constexpr uint64_t EXPANSION_COUNT_NOT_PADDED = INNER_CELLS;

}    // namespace fmm
}    // namespace octotiger
