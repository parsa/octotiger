#include <cstdlib>
#include <iostream>
#include <string>

#include "../src/kernels/interactions_constants.hpp"
#include "../src/kernels/m2m_interactions.hpp"

namespace test_padded_grid_indexing {

void test_padded_grid_indexingother_kernel_directory() {
    // std::cerr << "error: changing the kernel directory lead to non-functional "
    //              "kernel: "
    //           << __FILE__ << " " << __LINE__ << std::endl;
    // exit(1);
}

void test_padded_grid_indexing() {
    std::cout << "DIMENSION: " << octotiger::fmm::DIMENSION << std::endl;
    std::cout << "INNER_CELLS_PER_DIRECTION: " << octotiger::fmm::INNER_CELLS_PER_DIRECTION
              << std::endl;
    std::cout << "INNER_CELLS: " << octotiger::fmm::INNER_CELLS << std::endl;
    std::cout << "INNER_CELLS_PADDING_DEPTH: " << octotiger::fmm::INNER_CELLS_PADDING_DEPTH
              << std::endl;
    // std::cout << "OUTER_CELLS_PER_DIRECTION: " << octotiger::fmm::OUTER_CELLS_PER_DIRECTION
    //           << std::endl;
    // std::cout << "OUTER_CELLS: " << octotiger::fmm::OUTER_CELLS << std::endl;

    std::cout << "PADDING_EDGE_SURFACE_SINGLE_DIRECTION: "
              << octotiger::fmm::PADDING_EDGE_SURFACE_SINGLE_DIRECTION << std::endl;
    std::cout << "PADDED_STRIDE: " << octotiger::fmm::PADDED_STRIDE << std::endl;
    std::cout << "VERTICES: " << octotiger::fmm::VERTICES << std::endl;

    std::cout << "EXPANSION_COUNT_PADDING: " << octotiger::fmm::EXPANSION_COUNT_PADDING
              << std::endl;
    std::cout << "EXPANSION_COUNT_NOT_PADDED: " << octotiger::fmm::EXPANSION_COUNT_NOT_PADDED << std::endl;
    std::cout << "EXPANSION_COUNT_PADDED: " << octotiger::fmm::EXPANSION_COUNT_PADDED << std::endl;


    // other_kernel_directory();
}

void test_construct_interactions_kernel() {
    octotiger::fmm::m2m_interactions();
}
}

int main(void) {
    std::cout << "info: test exits with an error message if any error occurs" << std::endl;
    test_padded_grid_indexing::test_padded_grid_indexing();
    test_padded_grid_indexing::test_construct_interactions_kernel();
    std::cout << "no early exit, tests ran successfully" << std::endl;
}
