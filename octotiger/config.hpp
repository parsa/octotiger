//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_CONFIG_HPP
#define OCTOTIGER_CONFIG_HPP

namespace octotiger
{
    typedef int integer;
    typedef double real;

    constexpr integer NDIM = 3;
    constexpr integer XDIM = 0;
    constexpr integer YDIM = 1;
    constexpr integer ZDIM = 2;

    const integer NFACE = 2 * NDIM;
    const integer NVERTEX = 8;
    constexpr integer NCHILD = 8;
}

#endif
