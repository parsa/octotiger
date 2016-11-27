//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_GEOMETRY_FACE_HPP
#define OCTOTIGER_GEOMETRY_FACE_HPP

#include <octotiger/config.hpp>

#include <hpx/include/serialization.hpp>

#include <array>

namespace octotiger { namespace geometry {
    struct face
    {
        static constexpr integer count = 6;

        integer value_;

        constexpr face() : value_(-1) {}
        explicit constexpr face(integer value) : value_(value) {}

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

        static constexpr std::array<face, count> full_set()
        {
            return
                {{face(0), face(1), face(2),
                  face(3), face(4), face(5)}};
        }
    };
}}

HPX_IS_BITWISE_SERIALIZABLE(octotiger::geometry::face);

#endif
