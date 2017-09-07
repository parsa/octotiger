#pragma once

#include "interaction_constants.hpp"
#include "struct_of_array_data.hpp"
// #include "taylor.hpp"
//
// #include "cuda_runtime.h"
//
namespace octotiger {
namespace fmm {
    namespace cuda {

        class m2m_cuda
        {
        public:
            // m2m_cuda(cudaStream_t stream) {
            //     stream_ = stream;
            // }

            void compute_interactions(
                octotiger::fmm::struct_of_array_data<real, 20, ENTRIES_PADDED, SOA_PADDING>&
                    local_expansions_SoA,
                octotiger::fmm::struct_of_array_data<real, 3, ENTRIES_PADDED, SOA_PADDING>&
                    center_of_masses_SoA,
                octotiger::fmm::struct_of_array_data<real, 20, ENTRIES_NOT_PADDED, SOA_PADDING>&
                    potential_expansions_SoA,
                octotiger::fmm::struct_of_array_data<real, 3, ENTRIES_NOT_PADDED, SOA_PADDING>&
                    angular_corrections_SoA,
                double theta, std::array<real, 20>& factor, std::array<real, 20>& factor_half,
                std::array<real, 20>& factor_sixth);
            // private:
            //   cudaStream_t stream_;
        };
    }
}
}
