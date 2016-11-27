//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <octotiger/tree_location.hpp>

#include <octotiger/geometry/constants.hpp>
#include <octotiger/geometry/octant.hpp>

#include <hpx/util/assert.hpp>

namespace octotiger
{
    tree_location::tree_location(integer max_level)
      : loc_({0})
      , level_(0)
      , max_level_(max_level)
    {}

    integer tree_location::index() const
    {
        if (level_ == 0)
            return 0;

        integer increment = std::pow(8, max_level_ - level_);

        return child_index() * increment + parent().index() + max_level_ - level_;
    }

    tree_location tree_location::child(integer index) const
    {
	    integer x = (index >> 0) & 1;
        integer y = (index >> 1) & 1;
        integer z = (index >> 2) & 1;

        tree_location child(max_level_);
        child.level_ = level_ + 1;
        child.loc_[XDIM] = 2 * loc_[XDIM] + x;
        child.loc_[YDIM] = 2 * loc_[YDIM] + y;
        child.loc_[ZDIM] = 2 * loc_[ZDIM] + z;
        return child;
    }

    tree_location tree_location::parent() const
    {
        HPX_ASSERT(level_ >= 1);

        tree_location parent(max_level_);
        parent.level_ = level_ - 1;
        for (integer d = 0; d != NDIM; ++d) {
            parent.loc_[d] = loc_[d] / 2;
        }
        return parent;
    }

    tree_location tree_location::sibling(integer face) const
    {
        tree_location sibling(*this);
        const integer dim = face / 2;
        const integer dir = face % 2;
        if (dir == 0) {
            sibling.loc_[dim]--;
            HPX_ASSERT(sibling.loc_[dim] >= 0);
        }
        else {
            sibling.loc_[dim]++;
            assert(sibling.loc_[dim] < (1 << level_));
        }
        return sibling;
    }

    geometry::side tree_location::child_side(geometry::dimension const& dim) const
    {
        return geometry::side{(loc_[*dim] & 1) ? geometry::PLUS : geometry::MINUS};
    }

    integer tree_location::child_index() const
    {
        return
            *geometry::octant{
                std::array<geometry::side, NDIM>{
                    {
                        child_side(geometry::dimension{XDIM}),
                        child_side(geometry::dimension{YDIM}),
                        child_side(geometry::dimension{ZDIM})
                    }
                }
            };
    }

    template <typename Archive>
    void tree_location::serialize(Archive& ar, unsigned)
    {
        ar & loc_;
        ar & level_;
        ar & max_level_;
    }

    template void tree_location::serialize(hpx::serialization::input_archive&, unsigned);
    template void tree_location::serialize(hpx::serialization::output_archive&, unsigned);

    std::ostream& operator<<(std::ostream& os, tree_location const& loc)
    {
        os
            << "{ level = " << loc.level_
            << "; x = " << loc.loc_[XDIM]
            << "; y = " << loc.loc_[YDIM]
            << "; z = " << loc.loc_[ZDIM] << "; }";
        return os;
    }
}
