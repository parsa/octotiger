//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <octotiger/node_tree.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

// This unit test checks that we are able to correctly form a full octree

int main()
{
    using octotiger::integer;
    integer max_levels = 7;
    integer num_nodes = 1;
    for (integer max_level = 1; max_level <= max_levels; ++max_level)
    {
        std::cout << "starting with level " << max_level << "\n";
        num_nodes += std::pow(8, max_level);
        hpx::util::high_resolution_timer timer;
        int i = 0;
        for (i = 0; i < 10; ++i)
        {
            octotiger::node_tree tree(hpx::invalid_id, octotiger::tree_location{max_level});
            for (integer j = 0; j < max_level; ++j)
            {
                tree.regrid().get();
            }
            std::cout << "." << std::flush;
            HPX_TEST_EQ(tree.num_nodes(), num_nodes);
            if (timer.elapsed() > 20) break;
        }

        double elapsed = timer.elapsed();
        std::cout << max_level << " " << num_nodes << " " << elapsed/(i + 1) << '\n';
    }
    return hpx::util::report_errors();
}
