#pragma once

#include "multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

    std::array<std::vector<multiindex>, 8> calculate_stencil();

}    // namespace fmm
}    // namespace octotiger
