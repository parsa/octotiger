//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_GEOMETRY_OCTANT_HPP
#define OCTOTIGER_GEOMETRY_OCTANT_HPP

#include <octotiger/config.hpp>
#include <octotiger/geometry/constants.hpp>
#include <octotiger/geometry/face.hpp>
#include <octotiger/geometry/side.hpp>

#include <hpx/include/serialization.hpp>

#include <array>

namespace octotiger { namespace geometry {
    struct octant
    {
        static constexpr integer count = 8;

        integer value_;

        constexpr octant() : value_(-1) {}
        explicit constexpr octant(integer value) : value_(value) {}

        explicit constexpr octant(std::array<side, NDIM> const& sides)
          : value_(*sides[XDIM] | (*sides[YDIM] << 1) | (*sides[ZDIM] << 2))
        {
        }

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

        static constexpr std::array<octant, count> full_set()
        {
            return
                {{octant(0), octant(1), octant(2), octant(3),
                  octant(4), octant(5), octant(6), octant(7)}};
        }

        static std::array<octant, count/2> face_subset(face f)
        {
            switch (*f)
            {
                case FXM:
                    return {{octant(0) ,octant(2) ,octant(4) ,octant(6)}};
                case FXP:
                    return {{octant(1) ,octant(3) ,octant(5) ,octant(7)}};
		        case FYM:
                    return {{octant(0) ,octant(1) ,octant(4) ,octant(5)}};
                case FYP:
                    return {{octant(2) ,octant(3) ,octant(6) ,octant(7)}};
                case FZM:
                    return {{octant(0) ,octant(1) ,octant(2) ,octant(3)}};
                case FZP:
                    return {{octant(4) ,octant(5) ,octant(6) ,octant(7)}};
                default:
                    HPX_ASSERT(false);
            }

            return {};
        }
    };
}}

HPX_IS_BITWISE_SERIALIZABLE(octotiger::geometry::octant);

#endif
