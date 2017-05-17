#pragma once

#include "space_vector.hpp"
#include "simd.hpp"

#include <vector>

struct interaction_type
{
    // linear index in cell of first interaction partner
    std::uint16_t first;
    // linear index in cell of second interaction partner
    // (in case of pointing to neighbor, inner linear index in neighbor cell)
    std::uint16_t second;
    // index vector in cell
    space_vector x;
    // precomputed values: (-1.0/r, (i0-j0)/r^3, (i1-j1)/r^3, (i2-j2)/r^3), r - distance(i - j)
    v4sd four;
    // helper variable for vectorization
    std::uint32_t inner_loop_stop;
};

struct boundary_interaction_type
{
    // all interaction partners, if first.size() == 1, else the current index
    std::vector<std::uint16_t> second;
    // all interaction partners, if second.size() == 1, else the current index
    std::vector<std::uint16_t> first;
    // precomputed values, as in interaction_type
    std::vector<v4sd> four;
    // index vector in cell
    space_vector x;
};
