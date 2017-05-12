#include "multiindex.hpp"

std::ostream & operator<<(std::ostream &os, const octotiger::fmm::multiindex& m)
{
    return os << "x: " << m.x << " y: " << m.y << " z: " << m.z;
}
