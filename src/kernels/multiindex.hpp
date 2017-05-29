#pragma once

#include "defs.hpp"

#include <cmath>
#include <iostream>

namespace octotiger {
namespace fmm {

    template<typename T = int64_t>
    class multiindex
    {
    public:
        T x;
        T y;
        T z;

        multiindex(T x, T y, T z)
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

        // set this multiindex to the next coarser level index
        void transform_coarse() {
            x = (x + INX) / 2 - INX / 2;
            y = (y + INX) / 2 - INX / 2;
            z = (z + INX) / 2 - INX / 2;
        }
    };

}    // namespace fmm
}    // namespace octotiger

template<typename T>
std::ostream& operator<<(std::ostream& os, const octotiger::fmm::multiindex<T>& m) {
    return os << m.x << ", " << m.y << ", " << m.z;
}
