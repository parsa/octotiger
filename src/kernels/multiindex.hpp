#pragma once

#include <iostream>

namespace octotiger {
namespace fmm {

    struct multiindex
    {
    public:
        size_t x;
        size_t y;
        size_t z;

        multiindex(size_t x, size_t y, size_t z)
          : x(x)
          , y(y)
          , z(z) {}
    };

}    // namespace fmm
}    // namespace octotiger

std::ostream & operator<<(std::ostream &os, const octotiger::fmm::multiindex& m);
