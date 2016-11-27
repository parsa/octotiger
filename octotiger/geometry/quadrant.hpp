//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_GEOMETRY_QUADRANT_HPP
#define OCTOTIGER_GEOMETRY_QUADRANT_HPP

#include <octotiger/config.hpp>

#include <hpx/include/serialization.hpp>

#include <array>

namespace octotiger { namespace geometry {
    struct quadrant
    {
        static constexpr integer count = 4;

        integer value_;

        constexpr quadrant() : value_(-1) {}
        explicit constexpr quadrant(integer value) : value_(value) {}

        constexpr integer operator*() const
        {
//             HPX_ASSERT(value_ != -1);
            return value_;
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & value_;
            HPX_ASSERT(value_ != -1);
        }

        static constexpr std::array<quadrant, count> full_set()
        {
            return
                {{quadrant(0), quadrant(1), quadrant(2), quadrant(3)}};
        }
    };
}}

HPX_IS_BITWISE_SERIALIZABLE(octotiger::geometry::quadrant);

#endif
