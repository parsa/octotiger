//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCTOTIGER_NODE_TREE_COMPONENT_HPP
#define OCTOTIGER_NODE_TREE_COMPONENT_HPP

#include <octotiger/geometry/face.hpp>
#include <octotiger/node_tree.hpp>
#include <octotiger/tree_location.hpp>

#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/serialization.hpp>

#include <vector>

namespace octotiger
{
    class node_tree_component
      : public hpx::components::managed_component_base<node_tree_component>
    {
    public:
        node_tree_component();
        node_tree_component(node_tree_component const& other);
        node_tree_component(
            hpx::naming::id_type&& parent,
            tree_location&& location
        );

        integer num_nodes();
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, num_nodes);

        hpx::future<void> regrid();
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, regrid);

        void regrid_scatter(integer index, integer total);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, regrid_scatter);

        void regrid_gather(integer child, integer num_subnodes);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, regrid_gather);

        hpx::future<void> form_tree(std::vector<node_tree> neighbors);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, form_tree);

        void set_aunt(hpx::id_type aunt, geometry::face face);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, set_aunt);

        node_tree get_child(integer index);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, get_child);

        std::vector<node_tree> get_nieces(hpx::id_type aunt, geometry::face face);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, get_nieces);

        node_tree copy(hpx::id_type destination);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, copy);

        hpx::future<void> update_children();
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, update_children);
        void set_parent(hpx::id_type parent);
        HPX_DEFINE_COMPONENT_ACTION(node_tree_component, set_parent);

    private:
        tree_location location_;
        std::array<integer, NCHILD> num_subnodes_;

        std::vector<node_tree> children_;
        std::vector<node_tree> neighbors_;
        std::vector<node_tree> siblings_;
        std::vector<std::vector<node_tree>> nieces_;
        std::vector<node_tree> aunts_;
        node_tree parent_;

        // Managing regridding...
        hpx::future<integer> check_refinement();
        void refine(bool needs_refinement, integer index, integer total);
        void clear_family();

        std::vector<hpx::lcos::local::channel<integer>> num_subnodes_channel_;
        hpx::lcos::local::channel<std::pair<integer, integer>> total_nodes_channel_;

        friend class hpx::serialization::access;
        void load(hpx::serialization::input_archive& ar, unsigned);
        void save(hpx::serialization::output_archive& ar, unsigned) const;
        HPX_SERIALIZATION_SPLIT_MEMBER();
    };
}

#endif
