#pragma once

#include <iostream>
#include <cmath>

namespace octotiger {
namespace fmm {

    struct multiindex
    {
    public:
        int64_t x;
        int64_t y;
        int64_t z;

        multiindex(size_t x, size_t y, size_t z)
          : x(x)
          , y(y)
          , z(z) {}

        inline const double length() const {
            return sqrt(static_cast<double>(x * x + y * y + z * z));
        }
    };

}    // namespace fmm
}    // namespace octotiger

std::ostream & operator<<(std::ostream &os, const octotiger::fmm::multiindex& m);
