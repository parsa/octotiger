//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_NODE_STATE_HPP
#define OCTOTIGER_NODE_STATE_HPP

namespace octotiger
{
    struct node_state
    {
        enum program_state
        {
            state_init,
            state_run,
            state_done,
        };

        enum computation_state
        {
            state_form_tree,
            state_regrid_gather,
            state_regrid_scatter,
            state_solve_gravity,

        };
    };
}

#endif
