//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_GEOMETRY_DIRECTION_HPP
#define OCTOTIGER_GEOMETRY_DIRECTION_HPP

#include <octotiger/config.hpp>
#include <octotiger/geometry/constants.hpp>
#include <octotiger/geometry/face.hpp>

#include <hpx/include/serialization.hpp>

#include <array>

namespace octotiger { namespace geometry {
    struct direction
    {
        static constexpr integer count = 26;

        integer value_;

        constexpr direction() : value_(-1) {}
        explicit constexpr direction(integer value) : value_(value) {}
        constexpr direction(integer x, integer y, integer z)
          : value_(
                ((x == 0) ? 1 : (x > 0) ? 2 : 0) +
                ((y == 0) ? NDIM : (y > 0) ? 2 * NDIM : 0) +
                ((z == 0) ? NDIM * NDIM : (z > 0) ? 2 * NDIM * NDIM : 0)
            )
        {
        }

        explicit direction(face f)
        {
            switch (*f)
            {
                case FXM:
                    value_ = *direction(-1, 0, 0);
                    break;
                case FXP:
                    value_ = *direction(1, 0, 0);
                    break;
                case FYM:
                    value_ = *direction(0, -1, 0);
                    break;
                case FYP:
                    value_ = *direction(0, 1, 0);
                    break;
                case FZM:
                    value_ = *direction(0, 0, -1);
                    break;
                case FZP:
                    value_ = *direction(0, 0, 1);
                    break;
                default:
                    value_ = -1;
            }
        }

        constexpr integer operator*() const
        {
//             HPX_ASSERT(value_ != -1);
            return (value_ < 14) ? value_ : (value_ - 1);
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            ar & value_;
            HPX_ASSERT(value_ != -1);
        }

        static constexpr std::array<direction, count> full_set()
        {
            return
                {{direction(0), direction(1), direction(2), direction(3),
                  direction(4), direction(5), direction(6), direction(7),
                  direction(8), direction(9), direction(10), direction(11),
                  direction(12), /*direction(13),*/ direction(14), direction(15),
                  direction(16), direction(17), direction(18), direction(19),
                  direction(20), direction(21), direction(22), direction(23),
                  direction(24), direction(25), direction(26)}};
        }
    };
}}

HPX_IS_BITWISE_SERIALIZABLE(octotiger::geometry::direction);

#endif
