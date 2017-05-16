#include "multiindex.hpp"

std::ostream & operator<<(std::ostream &os, const octotiger::fmm::multiindex& m)
{
    return os << m.x << ", " << m.y << ", " << m.z;
}
