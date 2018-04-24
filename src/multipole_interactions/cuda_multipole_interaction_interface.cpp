#ifdef OCTOTIGER_CUDA_ENABLED
#include "cuda_multipole_interaction_interface.hpp"
#include "calculate_stencil.hpp"
#include "multipole_cuda_kernel.hpp"
#include "options.hpp"

extern options opts;
namespace octotiger {
namespace fmm {
    namespace multipole_interactions {

        cuda_multipole_interaction_interface::cuda_multipole_interaction_interface(void)
          : multipole_interaction_interface()
          , theta(opts.theta) {}

        void cuda_multipole_interaction_interface::compute_multipole_interactions(
            std::vector<real>& monopoles, std::vector<multipole>& M_ptr,
            std::vector<std::shared_ptr<std::vector<space_vector>>>& com_ptr,
            std::vector<neighbor_gravity_type>& neighbors, gsolve_type type, real dx,
            std::array<bool, geo::direction::count()>& is_direction_empty,
            std::array<real, NDIM> xbase) {
            // Check where we want to run this:
            int slot = kernel_scheduler::scheduler.get_launch_slot();
            if (slot == -1) {    // Run fkernel_scheduler::allback cpu implementation
                // std::cout << "Running cpu fallback" << std::endl;
                multipole_interaction_interface::compute_multipole_interactions(
                    monopoles, M_ptr, com_ptr, neighbors, type, dx, is_direction_empty, xbase);
            } else {    // run on cuda device
                // std::cerr << "Running cuda in slot " << slot << std::endl;
                // Move data into SoA arrays
                auto staging_area = kernel_scheduler::scheduler.get_staging_area(slot);
                update_input(monopoles, M_ptr, com_ptr, neighbors, type, dx, xbase,
                    staging_area.local_monopoles, staging_area.local_expansions_SoA,
                    staging_area.center_of_masses_SoA);

                // Queue moving of input data to device
                util::cuda_helper& gpu_interface =
                    kernel_scheduler::scheduler.get_launch_interface(slot);
                kernel_device_enviroment& env =
                    kernel_scheduler::scheduler.get_device_enviroment(slot);
                gpu_interface.copy_async(env.device_local_monopoles,
                    staging_area.local_monopoles.data(), local_monopoles_size,
                    cudaMemcpyHostToDevice);
                gpu_interface.copy_async(env.device_local_expansions,
                    staging_area.local_expansions_SoA.get_pod(), local_expansions_size,
                    cudaMemcpyHostToDevice);
                gpu_interface.copy_async(env.device_center_of_masses,
                    staging_area.center_of_masses_SoA.get_pod(), center_of_masses_size,
                    cudaMemcpyHostToDevice);

                // Launch kernel and queue copying of results
                const dim3 grid_spec(2);
                const dim3 threads_per_block(4, 4, 8);
                const dim3 sum_spec(1);
                const dim3 threads(128);

                std::array<hpx::future<void>, 4> compute_futs;
                for (auto x = 0; x < 2; ++x) {
                  for (auto y = 0; y < 2; ++y) {
                size_t index = x * 2 + y;
                // if (index != 0) {
                //   slot = kernel_scheduler::scheduler.get_launch_slot();
                //  gpu_interface = kernel_scheduler::scheduler.get_launch_interface(slot);
                //  env = kernel_scheduler::scheduler.get_device_enviroment(slot);
                // }
                if (type == RHO) {
                    void* args[] = {&(env.device_local_monopoles), &(env.device_center_of_masses),
                        &(env.device_local_expansions), &(env.device_potential_expansions),
                        &(env.device_angular_corrections), &(env.device_stencil),
                                    &(env.device_phase_indicator), &theta, &x, &y};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_rho, grid_spec,
                        threads_per_block, args, 0);
                    void* sum_args[] = {&(env.device_angular_corrections)};
                    gpu_interface.execute(
                        &cuda_add_multipole_ang_blocks, sum_spec, threads, sum_args, 0);
                    gpu_interface.copy_async(angular_corrections_SoA[index].get_pod(),
                        env.device_angular_corrections, angular_corrections_size,
                        cudaMemcpyDeviceToHost);

                } else {
                    void* args[] = {&(env.device_local_monopoles), &(env.device_center_of_masses),
                        &(env.device_local_expansions), &(env.device_potential_expansions),
                        &(env.device_stencil), &(env.device_phase_indicator), &theta, &x, &y};
                    gpu_interface.execute(&cuda_multipole_interactions_kernel_non_rho, grid_spec,
                        threads_per_block, args, 0);
                }
                void* sum_args[] = {&(env.device_potential_expansions)};
                gpu_interface.execute(
                    &cuda_add_multipole_pot_blocks, sum_spec, threads, sum_args, 0);
                gpu_interface.copy_async(potential_expansions_SoA[index].get_pod(),
                    env.device_potential_expansions, potential_expansions_size,
                    cudaMemcpyDeviceToHost);

                // Wait for stream to finish and allow thread to jump away in
                // the meantime
                compute_futs[index] = gpu_interface.get_future().then([this, x, y, index, type](
                    hpx::future<void> &&f) {
                  // Copy results back into non-SoA array
                  std::vector<expansion> &L = grid_ptr->get_L();
                  std::vector<space_vector> &L_c = grid_ptr->get_L_c();
                  auto data_c = angular_corrections_SoA[index].get_pod();
                  auto data = potential_expansions_SoA[index].get_pod();

                  for (auto i0 = 0; i0 < COMPUTE_BLOCK_LENGTH; ++i0) {
                    for (auto i1 = 0; i1 < COMPUTE_BLOCK_LENGTH; ++i1) {
                      for (auto i2 = 0; i2 < INNER_CELLS_PER_DIRECTION; ++i2) {
                        const multiindex<> cell_index_unpadded(i0, i1, i2);
                        const int64_t cell_flat_index_unpadded =
                            i0 * COMPUTE_BLOCK_LENGTH *
                                INNER_CELLS_PER_DIRECTION +
                            i1 * INNER_CELLS_PER_DIRECTION + i2;
                        const multiindex<> cell_index(
                            i0 + x * COMPUTE_BLOCK_LENGTH,
                            i1 + y * COMPUTE_BLOCK_LENGTH, i2);
                        const int64_t cell_flat_index =
                            to_inner_flat_index_not_padded(cell_index);

                        for (size_t component = 0; component < 20;
                             component++) {
                          L[cell_flat_index][component] +=
                              data[component * (COMPUTE_BLOCK + SOA_PADDING) +
                                   cell_flat_index_unpadded];
                        }
                        for (size_t component = 0; component < 3 && type == RHO;
                             component++) {
                          L_c[cell_flat_index][component] +=
                              data_c[component * (COMPUTE_BLOCK + SOA_PADDING) +
                                     cell_flat_index_unpadded];
                        }
                      }
                    }
                  }
                });
                  }
                }
                auto fut = hpx::when_all(compute_futs);
                fut.get();
            }
        }

    }    // namespace multipole_interactions
}    // namespace fmm
}    // namespace octotiger
#endif
