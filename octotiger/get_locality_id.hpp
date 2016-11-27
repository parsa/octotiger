//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_GET_LOCALITY_ID_HPP
#define OCTOTIGER_GET_LOCALITY_ID_HPP

#include <octotiger/config.hpp>

#include <hpx/include/naming.hpp>

namespace octotiger
{
    hpx::id_type get_locality_id(integer index, integer total);
}

#endif
