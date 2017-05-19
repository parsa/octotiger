#pragma once

#include <cmath>
#include <iostream>

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

        inline bool compare(multiindex& other) {
            if (this->x == other.x && this->y == other.y && this->z == other.z) {
                return true;
            } else {
                return false;
            }
        }
    };

}    // namespace fmm
}    // namespace octotiger

std::ostream& operator<<(std::ostream& os, const octotiger::fmm::multiindex& m);
