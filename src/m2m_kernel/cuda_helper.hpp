#pragma once

#include <iostream>
#include "struct_of_array_data.hpp"

namespace octotiger {
  namespace fmm {
    namespace cuda {

      inline void check_cuda_error(cudaError_t error, const char *file, int line) {
	if (error != cudaSuccess) {
	  std::cerr << "CUDA error: " << cudaGetErrorString(error) << " file: " << file << " line: " << line << std::endl;
	}
      }
    }
  }
}

#define CUDA_CHECK_ERROR(error) { octotiger::fmm::cuda::check_cuda_error((error), __FILE__, __LINE__); }      
namespace octotiger {
  namespace fmm {
    namespace cuda {

    class cuda_buffer {
    private:
      double *d_ptr;
    public:
      cuda_buffer(double *d_ptr_): d_ptr(d_ptr_) {}
      ~cuda_buffer() {
	if (d_ptr != nullptr) {
	  CUDA_CHECK_ERROR(cudaFree(d_ptr));
	}
      }
      cuda_buffer(cuda_buffer &&other) {
	d_ptr = other.d_ptr;
	other.d_ptr = nullptr;
      };
      cuda_buffer(const cuda_buffer &) = delete;
      void operator=(const cuda_buffer& other) = delete;

      double *get_device_ptr() {
	return d_ptr;
      }
    };

      template <typename AoS_type, typename component_type, size_t num_components, size_t entries,
		size_t padding>
      cuda_buffer move_to_device(octotiger::fmm::struct_of_array_data<AoS_type, component_type, num_components, entries, padding> &data) {
	double *d_SoA;
	size_t SoA_bytes = data.get_size_bytes();
	CUDA_CHECK_ERROR(cudaMalloc(&d_SoA, SoA_bytes));
	double *h_SoA = data.get_underlying_pointer();
	CUDA_CHECK_ERROR(cudaMemcpy(d_SoA, h_SoA, SoA_bytes, cudaMemcpyHostToDevice));
	return cuda_buffer(d_SoA);
      }

    // component type has to be indexable, and has to have a size() operator
    template <typename AoS_type, typename component_type, size_t num_components, size_t entries,
        size_t padding>
    class struct_of_array_data
    {
    private:
        // data in SoA form
        static constexpr size_t padded_entries_per_component = entries + padding;

        component_type* const data;

    public:
        template <size_t component_access>
        inline component_type* pointer(const size_t flat_index) const {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            // should result in single move instruction, indirect addressing: reg + reg + constant
            return data + flat_index + component_array_offset;
        }

        // careful, this returns a copy!
        template <size_t component_access>
        inline m2m_vector value(const size_t flat_index) const {
            return m2m_vector(
                this->pointer<component_access>(flat_index), Vc::flags::element_aligned);
        }

        // template <size_t component_access>
        // inline component_type& reference(const size_t flat_index) const {
        //     return *this->pointer<component_access>(flat_index);
        // }

        __device__ struct_of_array_data(const std::vector<AoS_type>& org)
          : data(new component_type[num_components * padded_entries_per_component]) {
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    data[component * padded_entries_per_component + entry] = org[entry][component];
                }
            }
        }

        // constructor that works on preallocated and initialized data
        __device__ struct_of_array_data(component_type* preallocated_data)
	  : data(preallocated_data) {
        }

        __device__ struct_of_array_data(const size_t entries_per_component)
          : data(new component_type[num_components * padded_entries_per_component]) {}

        __device__ ~struct_of_array_data() {
            delete[] data;
        }

        __device__ struct_of_array_data(const struct_of_array_data& other) = delete;

        __device__ struct_of_array_data(const struct_of_array_data&& other) = delete;

        __device__ struct_of_array_data& operator=(const struct_of_array_data& other) = delete;

        // write back into non-SoA style array
        __device__ void to_non_SoA(std::vector<AoS_type>& org) {
            // constexpr size_t padded_entries_per_component = entries + padding;
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    org[entry][component] = data[component * padded_entries_per_component + entry];
                }
            }
        }

      __device__ double *get_underlying_pointer() {
	return data;
      }

      __device__ size_t get_size_bytes() {
	return num_components * padded_entries_per_component * sizeof(component_type);
      }
    };

    }
  }
}
