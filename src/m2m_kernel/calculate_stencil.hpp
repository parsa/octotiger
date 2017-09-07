#pragma once

#include "multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

    extern std::vector<octotiger::fmm::multiindex<>> stencil;

    std::vector<multiindex<>> calculate_stencil();

}    // namespace fmm
}    // namespace octotiger
