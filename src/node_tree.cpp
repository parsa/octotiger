//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <octotiger/node_tree.hpp>
#include <octotiger/node_tree_component.hpp>

namespace octotiger
{
    node_tree::node_tree()
      : base_type(hpx::invalid_id)
    {}

    node_tree::node_tree(hpx::naming::id_type parent, tree_location&& location)
      : base_type(hpx::new_<node_tree_component>(hpx::find_here(), parent, std::move(location)))
    {
    }

    hpx::future<void> node_tree::regrid()
    {
        return hpx::async<node_tree_component::regrid_action>(this->get_id());
    }

    void node_tree::regrid_scatter(integer index, integer total)
    {
        hpx::apply<node_tree_component::regrid_scatter_action>(this->get_id(), index, total);
    }

    void node_tree::regrid_gather(integer child, integer num_subnodes)
    {
        hpx::apply<node_tree_component::regrid_gather_action>(
            this->get_id(), child, num_subnodes);
    }

    hpx::future<void> node_tree::form_tree(std::vector<node_tree> neighbors)
    {
        return hpx::async<node_tree_component::form_tree_action>(this->get_id(), std::move(neighbors));
    }

    hpx::future<void> node_tree::set_aunt(hpx::id_type aunt, geometry::face face)
    {
        return hpx::async<node_tree_component::set_aunt_action>(this->get_id(), aunt, face);
    }

    node_tree node_tree::get_child(integer index)
    {
        return node_tree(hpx::async<node_tree_component::get_child_action>(this->get_id(), index));
    }

    hpx::future<std::vector<node_tree>> node_tree::get_nieces(hpx::id_type aunt, geometry::face face)
    {
        return hpx::async<node_tree_component::get_nieces_action>(this->get_id(), aunt, face);
    }

    node_tree node_tree::copy(hpx::id_type destination)
    {
        return node_tree(hpx::async<node_tree_component::copy_action>(this->get_id(), destination));
    }

    hpx::future<void> node_tree::set_parent(hpx::id_type parent)
    {
        return hpx::async<node_tree_component::set_parent_action>(this->get_id(), parent);
    }

    void node_tree::update_children()
    {
        node_tree_component::update_children_action()(this->get_id()).get();
    }

    integer node_tree::num_nodes()
    {
        return node_tree_component::num_nodes_action()(this->get_id());
    }
}
