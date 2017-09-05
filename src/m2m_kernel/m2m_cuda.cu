#include "m2m_cuda.hpp"

#include "struct_of_array_data.hpp"

__global__ void trivial_kernel(float *f) {
  printf("hello from gpu");
}

void test(cudaStream_t stream) {
  float *d;
  cudaMalloc(&d, 128 * sizeof(float));
  trivial_kernel<<<1, 1, 0, stream>>>(d);
  cudaFree(d);
}

namespace octotiger {
namespace fmm {
namespace cuda {

void m2m_cuda::compute_interactions(
    octotiger::fmm::struct_of_array_data<expansion, real, 20, ENTRIES,
                                         SOA_PADDING> &local_expansions_SoA,
    octotiger::fmm::struct_of_array_data<space_vector, real, 3, ENTRIES,
                                         SOA_PADDING> &center_of_masses_SoA,
    octotiger::fmm::struct_of_array_data<expansion, real, 20, ENTRIES,
                                         SOA_PADDING> &potential_expansions_SoA,
    octotiger::fmm::struct_of_array_data<
        space_vector, real, 3, ENTRIES, SOA_PADDING> &angular_corrections_SoA)
    {
      test(stream_);
    }
}
}
}
