//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_NODE_TREE_HPP
#define OCTOTIGER_NODE_TREE_HPP

#include <octotiger/geometry/face.hpp>
#include <octotiger/config.hpp>
#include <octotiger/tree_location.hpp>

#include <hpx/include/components.hpp>

#include <cstddef>

namespace octotiger
{
    class node_tree_component;
    class node_tree
        : public hpx::components::client_base<node_tree, node_tree_component>
    {
        typedef
            hpx::components::client_base<node_tree, node_tree_component>
            base_type;

    public:
        node_tree();
        explicit node_tree(hpx::naming::id_type id)
          : base_type(std::move(id))
        {}
        explicit node_tree(hpx::future<hpx::naming::id_type>&& id)
          : base_type(std::move(id))
        {}
        explicit node_tree(hpx::shared_future<hpx::naming::id_type>&& id)
          : base_type(std::move(id))
        {}
        explicit node_tree(hpx::shared_future<hpx::naming::id_type> const& id)
          : base_type(id)
        {}

        node_tree(hpx::future<node_tree>&& other)
          : base_type(std::move(other))
        {
        }

        node_tree(node_tree&& other)
          : base_type(static_cast<base_type&&>(std::move(other)))
        {}

        node_tree(node_tree const& other)
          : base_type(static_cast<base_type const&>(other))
        {}

        node_tree& operator=(hpx::id_type const& id)
        {
            base_type::operator=(id);
            return *this;
        }

        node_tree& operator=(hpx::id_type&& id)
        {
            base_type::operator=(std::move(id));
            return *this;
        }

        node_tree& operator=(hpx::shared_future<hpx::id_type> const& id)
        {
            base_type::operator=(id);
            return *this;
        }

        node_tree& operator=(hpx::shared_future<hpx::id_type>&& id)
        {
            base_type::operator=(std::move(id));
            return *this;
        }

        node_tree& operator=(hpx::future<hpx::id_type>&& id)
        {
            base_type::operator=(std::move(id));
            return *this;
        }

        node_tree& operator=(node_tree&& other)
        {
            base_type::operator=(static_cast<base_type&&>(std::move(other)));
            return *this;
        }

        node_tree& operator=(node_tree const& other)
        {
            base_type::operator=(static_cast<base_type const&>(other));
            return *this;
        }

        node_tree(hpx::naming::id_type parent, tree_location&& location);

        hpx::future<void> regrid();

        void regrid_scatter(integer index, integer total);

        hpx::future<void> form_tree();

        void regrid_gather(integer child, integer num_subnodes);

        hpx::future<void> form_tree(std::vector<node_tree> neighbors);

        hpx::future<void> set_aunt(hpx::id_type aunt, geometry::face face);

        node_tree get_child(integer index);

        hpx::future<std::vector<node_tree>> get_nieces(hpx::id_type aunt, geometry::face face);

        node_tree copy(hpx::id_type destination);
        void update_children();
        hpx::future<void> set_parent(hpx::id_type destination);

        integer num_nodes();
    };
}

#endif
