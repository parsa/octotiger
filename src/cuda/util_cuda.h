#ifndef UTIL_CUDA_H
#define UTIL_CUDA_H

#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace octotiger { namespace cuda { namespace util 
{

  void cuda_error(cudaError_t err) {
    if (err != cudaSuccess) {
      std::stringstream temp;
      temp << "cuda function returned error code " << cudaGetErrorString(err);
      throw std::runtime_error(temp.str());
    }
  }

  template <class T>
  bool isGpuPointer(const T* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return attributes.memoryType == cudaMemoryTypeDevice;
  }
}}}

#endif
