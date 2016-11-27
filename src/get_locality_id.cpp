//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <octotiger/get_locality_id.hpp>

#include <hpx/include/runtime.hpp>

#include <vector>

namespace octotiger
{
    std::vector<hpx::id_type> const& get_localities()
    {
        static std::vector<hpx::id_type> localities(hpx::find_all_localities());

        return localities;
    }

    hpx::id_type get_locality_id(integer index, integer total)
    {
        auto const& localities = get_localities();
        integer loc_index = index * localities.size() / total;
        HPX_ASSERT(loc_index < localities.size());
        return localities[loc_index];
    }
}
