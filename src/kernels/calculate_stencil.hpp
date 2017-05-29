#pragma once

#include "m2m_parameters.hpp"
#include "multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

#ifdef M2M_SUPERIMPOSED_STENCIL
    std::vector<multiindex<>> calculate_stencil();
#else
    std::array<std::vector<multiindex<>>, 8> calculate_stencil();
#endif

}    // namespace fmm
}    // namespace octotiger
