#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

//#ifdef WITH_CUDA

#include "cuda_runtime.h"
#include "hpx/hpx.hpp"
#include <mutex>
#include <hpx/include/compute.hpp>
#include <hpx/compute/cuda/target.hpp>
#include "util.h"
#include "util_cuda.h"

// -------------------------------------------------------------------------
// a simple cuda wrapper helper object that can be used to synchronize
// calls with an hpx future.
// -------------------------------------------------------------------------
namespace octotiger { namespace cuda { namespace util 
{

    struct cuda_helper {
      public:
        using future_type = hpx::future<void>;
        using MutexType = hpx::lcos::local::spinlock;

        // construct a cuda stream
        cuda_helper(std::size_t device = 0) : target_(device) {
            stream_ = target_.native_handle().get_stream();
        }

        ~cuda_helper() {
        }

        cuda_helper(const cuda_helper&) = delete;
        cuda_helper(cuda_helper&& rhs)
           : stream_(rhs.stream_), target_(std::move(rhs.target_)) 
        {
           rhs.stream_ = nullptr;
        }

        cuda_helper& operator=(const cuda_helper&) = delete;
        cuda_helper& operator=(cuda_helper&& rhs) {
            stream_ = rhs.stream_;
            target_ = std::move(rhs.target_);
            rhs.stream_ = nullptr;
            return *this;
        }

        cudaStream_t get_stream() {
            return stream_;
        }

      // This is a simple wrapper for any user defined call, pass in the same arguments
      // that you would use for the asynchcronous call except the cuda stream
      // (last argument) which is omitted as the wrapper will supply that for you.
      template <typename Func, typename... Args>
      void execute(Func&& cuda_function, Args&&... args) {
        std::lock_guard<MutexType> lock(get_cuda_mutex());
        // make sure we run on the correct device
        util::cuda_error(cudaSetDevice(target_.native_handle().get_device()));

        // insert the cuda stream in the arg list and call the cuda async function
        return cuda_function(std::forward<Args>(args)..., stream_);
      }

      // This is a simple wrapper for any cuda call, pass in the same arguments
      // that you would use for a cuda asynchcronous call except the cuda stream
      // (last argument) which is omitted as the wrapper will supply that for you.
      template <typename Func, typename... Args>
      void execute_cuda(Func&& cuda_function, Args&&... args) {
        std::lock_guard<MutexType> lock(get_cuda_mutex());
        // make sure we run on the correct device
        util::cuda_error(cudaSetDevice(target_.native_handle().get_device()));

        // insert the cuda stream in the arg list and call the cuda async function
        util::cuda_error(cuda_function(std::forward<Args>(args)..., stream_));
      }

      // get the future to synchronize this cuda stream with
      future_type get_future() {
        return target_.get_future();
      }

      // return a reference to the compute::cuda object owned by this class
      hpx::compute::cuda::target& target() {
        return target_;
      }

      private:
      static MutexType& get_cuda_mutex() {
        static MutexType cuda_mtx_;
        return cuda_mtx_;
      }

      cudaStream_t stream_;
      hpx::compute::cuda::target target_;
    };

}}} // namespace

//#endif

#endif
