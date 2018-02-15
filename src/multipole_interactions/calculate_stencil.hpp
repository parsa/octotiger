#pragma once

#include "../common_kernel/multiindex.hpp"

#include <vector>

namespace octotiger {
namespace fmm {
    extern std::vector<octotiger::fmm::multiindex<>> stencil;

    std::vector<multiindex<>> calculate_stencil();
    namespace multipole_interactions {

        two_phase_stencil calculate_stencil(void);

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
