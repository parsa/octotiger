//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_TREE_LOCATION_HPP
#define OCTOTIGER_TREE_LOCATION_HPP

#include <octotiger/config.hpp>
#include <octotiger/geometry/dimension.hpp>
#include <octotiger/geometry/side.hpp>

#include <hpx/include/serialization.hpp>

#include <array>
#include <cstddef>
#include <iostream>

namespace octotiger
{
    class tree_location
    {
    public:
        tree_location() {}
        tree_location(integer max_level);

        integer level() const { return level_; }
        integer max_level() const { return max_level_; }

        integer index() const;

        tree_location child(integer index) const;
        tree_location parent() const;
        tree_location sibling(integer face) const;

        geometry::side child_side(geometry::dimension const& dim) const;
        integer child_index() const;

        bool is_physical_boundary() const;

    private:
        std::array<integer, NDIM> loc_;
        integer level_;
        integer max_level_;

        friend class hpx::serialization::access;
        template <typename Archive>
        void serialize(Archive& ar, unsigned);

        friend std::ostream& operator<<(std::ostream& os, tree_location const&);
    };
}

HPX_IS_BITWISE_SERIALIZABLE(octotiger::tree_location);

#endif
