//  Copyright (c) 2016 Dominic Marcello
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <octotiger/node_tree_component.hpp>
#include <octotiger/get_locality_id.hpp>
#include <octotiger/node_tree.hpp>
#include <octotiger/geometry/direction.hpp>
#include <octotiger/geometry/face.hpp>
#include <octotiger/geometry/octant.hpp>
#include <octotiger/geometry/quadrant.hpp>

#include <iostream>

typedef hpx::components::managed_component<octotiger::node_tree_component> node_tree_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(node_tree_component_type, ocototiger_node_tree);

namespace octotiger
{
    node_tree_component::node_tree_component()
    {
    }

    node_tree_component::node_tree_component(node_tree_component const& other)
      : location_(other.location_)
      , num_subnodes_(other.num_subnodes_)
      , children_(other.children_)
      , neighbors_(geometry::direction::count)
      , siblings_(NFACE)
      , nieces_(NFACE)
      , aunts_(NFACE)
      , parent_(other.parent_)
      , num_subnodes_channel_(other.num_subnodes_channel_.size())
    {
    }

    node_tree_component::node_tree_component(
        hpx::naming::id_type&& parent,
        tree_location&& location
    )
      : location_(std::move(location))
      , neighbors_(geometry::direction::count)
      , siblings_(NFACE)
      , nieces_(NFACE)
      , aunts_(NFACE)
      , parent_(std::move(parent))
    {
    }

    integer node_tree_component::num_nodes()
    {
        return std::accumulate(num_subnodes_.begin(), num_subnodes_.end(), 1);
    }

    hpx::future<void> node_tree_component::regrid()
    {
        std::vector<hpx::future<void>> regrid_futures;
        regrid_futures.reserve(children_.size());
        for (auto& child: children_)
        {
            regrid_futures.push_back(
                child.regrid()
            );
        }

        return hpx::dataflow(
            [this](
                hpx::future<integer>&& num_children_future,
                hpx::future<std::pair<integer, integer>>&& num_total_future,
                hpx::future<void>&& regrid_futures
            )
            {
                regrid_futures.get();
                auto num_children = num_children_future.get();
                auto num_total = num_total_future.get();

                HPX_ASSERT(num_children != 0);
                bool needs_refinement = num_children != 1;
                refine(needs_refinement, num_total.first, num_total.second);
                if (location_.level() == 0)
                {
                    HPX_ASSERT(num_total.first != -1);
                    return form_tree(neighbors_);
                }
                return hpx::make_ready_future();
            }
          , check_refinement()
          , total_nodes_channel_.get()
          , hpx::when_all(regrid_futures)
        );
    }

    void node_tree_component::regrid_scatter(integer index, integer total)
    {
        integer this_index = index;
        ++index;
        integer child_index = 0;
        for (auto& child: children_)
        {
            child.regrid_scatter(index, total);
            index += num_subnodes_[child_index];
            ++child_index;
        }
        total_nodes_channel_.set(std::make_pair(this_index, total));
    }

    void node_tree_component::regrid_gather(integer child, integer num_subnodes)
    {
        HPX_ASSERT(num_subnodes_channel_.size() > child);
        num_subnodes_channel_[child].set(num_subnodes);
    }

    void node_tree_component::set_aunt(hpx::id_type aunt, geometry::face f)
    {
        HPX_ASSERT(aunts_[*f].get_id() == hpx::invalid_id);
        aunts_[*f] = node_tree(aunt);
    }

    node_tree node_tree_component::get_child(integer index)
    {
        if (children_.empty())
            return node_tree(hpx::invalid_id);

        HPX_ASSERT(index < children_.size());
        return children_[index];
    }

    std::vector<node_tree> node_tree_component::get_nieces(hpx::id_type aunt, geometry::face face)
    {
        std::vector<node_tree> nieces;

        if (!children_.empty())
        {
            nieces.reserve(geometry::quadrant::count);
            for (auto& oct : geometry::octant::face_subset(face))
            {
                HPX_ASSERT(*oct < children_.size());
                node_tree niece = children_[*oct];
                HPX_ASSERT(niece.get_id());
                nieces.emplace_back(
                    niece.set_aunt(aunt, face).then(//hpx::launch::sync,
                        [niece](hpx::future<void> f)
                        {
                            // check error
                            f.get();
                            return niece.get_id();
                        }
                    )
                );
            }
        }

        return nieces;
    }

    node_tree node_tree_component::copy(hpx::id_type destination)
    {
        // Signal that this is done.
        total_nodes_channel_.set(std::make_pair(-1, 0));

        // TODO: copy grids etc.
        node_tree tree = hpx::new_<node_tree>(destination, *this);

        tree.update_children();

        return tree;
    }

    hpx::future<integer> node_tree_component::check_refinement()
    {
        std::vector<hpx::future<integer>> child_futures;
        child_futures.reserve(num_subnodes_channel_.size());
        for (auto& channel : num_subnodes_channel_)
        {
            child_futures.push_back(channel.get(hpx::launch::async));
        }

        // TODO: get hydro bounds

        bool refine_me = location_.level() < location_.max_level(); // TODO: check for refinement

        auto finalize = [this](integer num_subnodes)
        {
            if (num_subnodes == 1)
            {
                children_.clear();
                num_subnodes_channel_.clear();
            }
            // Inform our parent about our refinement results.
            if(parent_.get_id())
            {
                HPX_ASSERT(location_.level() != 0);
                parent_.regrid_gather(location_.child_index(), num_subnodes);
            }

            if (location_.level() == 0)
            {
                HPX_ASSERT(!parent_.get_id());
                // If we are at the root, we need to scatter the information about the
                // number of nodes and the indices.
                regrid_scatter(0, num_nodes());
            }
            else
            {
                HPX_ASSERT(parent_.get_id());
            }
            return num_subnodes;
        };

        if (child_futures.empty())
        {
            if (refine_me)
            {
                std::fill(num_subnodes_.begin(), num_subnodes_.end(), 1);
                return hpx::make_ready_future(finalize(NCHILD + 1));
            }
            else
            {
                std::fill(num_subnodes_.begin(), num_subnodes_.end(), 0);
                return hpx::make_ready_future(finalize(1));
            }
        }

        return
            hpx::dataflow(//hpx::launch::sync,
                [this, refine_me, finalize](std::vector<hpx::future<integer>> child_futures)
                {
                    integer num_subnodes = 1;

                    bool needs_refinement = refine_me;
                    integer index = 0;
                    for(hpx::future<integer>& child_future: child_futures)
                    {
                        integer num_children = child_future.get();

                        HPX_ASSERT(num_children != 0);
                        needs_refinement |= (num_children != 1);
                        num_subnodes += num_children;
                        num_subnodes_[index] = num_children;
                        ++index;
                    }

                    if (needs_refinement)
                        return finalize(num_subnodes);
                    else
                    {
                        std::fill(num_subnodes_.begin(), num_subnodes_.end(), 0);
                        return finalize(1);
                    }
                }
              , std::move(child_futures)
            );
    }

    hpx::future<void> node_tree_component::update_children()
    {
        std::vector<hpx::future<void>> update_futures;
        update_futures.reserve(children_.size());
        for (auto& child: children_)
        {
            update_futures.push_back(
                child.set_parent(this->get_id())
            );
        }

        return hpx::when_all(update_futures);
    }

    void node_tree_component::set_parent(hpx::id_type parent)
    {
        parent_ = parent;
    }

    void node_tree_component::refine(bool needs_refinement, integer index, integer total)
    {
        // If we got an index of -1, that means that the node got copied...
        if (index == -1)
            return;

        // If we have no children, we need to create them now...
        if (children_.empty() && needs_refinement)
        {
            num_subnodes_channel_.resize(NCHILD);
            children_.reserve(NCHILD);
//             ++index;
            for (auto& ci : geometry::octant::full_set())
            {
                HPX_ASSERT(location_.child(*ci).level() == location_.level() + 1);
                hpx::id_type locality = get_locality_id(index, total);
                node_tree new_node =
                    hpx::new_<node_tree>(
                        locality,
                        this->get_gid(),
                        location_.child(*ci)
                    );
                // TODO: set grid and outflow

                children_.push_back(new_node);
                ++index;
            }
        }
        else
        {
//             ++index;
            integer child_index = 0;
            for (auto& child: children_)
            {
                hpx::id_type destination = get_locality_id(index, total);
//                 HPX_ASSERT(destination == hpx::naming::get_locality_from_id(child.get_id()));
                if (destination != hpx::naming::get_locality_from_id(child.get_id()))
                {
                    // copy ...
                    child = child.copy(destination);
//                     regrid_futures.push_back(
//                         child.then(
//                             [regrid_steps](node_tree node)
//                             {
//                                 return node.regrid(regrid_steps);
//                             }
//                         )
//                     );
                }
                index += num_subnodes_[child_index];
                ++child_index;
            }
        }
    }

    void node_tree_component::clear_family()
    {
        std::fill(aunts_.begin(), aunts_.end(), node_tree{});
        std::fill(siblings_.begin(), siblings_.end(), node_tree{});
        std::fill(nieces_.begin(), nieces_.end(), std::vector<node_tree>{});
    }

    hpx::future<void> node_tree_component::form_tree(std::vector<node_tree> neighbors)
    {
        neighbors_ = neighbors;
        clear_family();

        HPX_ASSERT(neighbors_.size() == geometry::direction::count);

        for (auto face : geometry::face::full_set())
        {
            HPX_ASSERT(*geometry::direction(face) < neighbors_.size());
            HPX_ASSERT(*face < siblings_.size());
            HPX_ASSERT(siblings_[*face].get_id() == hpx::invalid_id);
            siblings_[*face] = neighbors_[*geometry::direction(face)];
        }

        if (!children_.empty())
        {
            std::vector<hpx::future<void>> neighbor_futures;
            neighbor_futures.reserve(children_.size());
            for (integer cx = 0; cx != 2; ++cx)
            {
                for (integer cy = 0; cy != 2; ++cy)
                {
                    for (integer cz = 0; cz != 2; ++cz)
                    {
                        const integer ci = cx + 2 * cy + 4 * cz;
                        HPX_ASSERT(ci < children_.size());
                        std::vector<node_tree> child_neighbors(geometry::direction::count);
                        for (integer dx = -1; dx != 2; ++dx)
                        {
                            for (integer dy = -1; dy != 2; ++dy)
                            {
                                for (integer dz = -1; dz != 2; ++dz)
                                {
                                    if (!(dx == 0 && dy == 0 && dz == 0))
                                    {
                                        const integer x = cx + dx + 2;
                                        const integer y = cy + dy + 2;
                                        const integer z = cz + dz + 2;
                                        geometry::direction d(dx, dy, dz);
                                        integer other_child = (x % 2) + 2 * (y % 2)
                                            + 4 * (z % 2);
                                        HPX_ASSERT(*d < child_neighbors.size());
                                        HPX_ASSERT(child_neighbors[*d].get_id() == hpx::invalid_id);
                                        if (x / 2 == 1 && y / 2 == 1 && z / 2 == 1)
                                        {
                                            HPX_ASSERT(other_child < children_.size());
                                            child_neighbors[*d] = children_[other_child];
                                        }
                                        else
                                        {
                                            geometry::direction dir(
                                                (x / 2) + NDIM * ((y / 2) + NDIM * (z / 2)));
                                            HPX_ASSERT(*dir < neighbors_.size());
                                            if (neighbors_[*dir].get_id())
                                            {
                                                child_neighbors[*d] = neighbors_[*dir].get_child(other_child);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // TODO: AMR flags

                        HPX_ASSERT(children_[ci].get_id());
                        neighbor_futures.push_back(
                            children_[ci].form_tree(std::move(child_neighbors))
                        );
                    }
                }
            }
            return hpx::when_all(neighbor_futures);
        }
        else
        {
            std::vector<hpx::future<void>> nieces_futures;
            nieces_futures.reserve(geometry::face::count);
            for (auto face: geometry::face::full_set())
            {
                if (siblings_[*face].get_id())
                {
                    nieces_futures.push_back(
                        siblings_[*face].get_nieces(this->get_id(), geometry::face(*face ^ 1)).then(//hpx::launch::sync,
                            [face, this](hpx::future<std::vector<node_tree>>&& nieces)
                            {
                                nieces_[*face] = nieces.get();
                            }
                        )
                    );
                }
            }
            return hpx::when_all(nieces_futures);
        }
    }

    void node_tree_component::load(hpx::serialization::input_archive& ar, unsigned)
    {
        ar & location_;
        ar & num_subnodes_;
        ar & children_;
        ar & parent_;
        neighbors_.resize(geometry::direction::count);
        siblings_.resize(NFACE);
        nieces_.resize(NFACE);
        aunts_.resize(NFACE);
        num_subnodes_channel_.resize(children_.size());
    }

    void node_tree_component::save(hpx::serialization::output_archive& ar, unsigned) const
    {
        ar & location_;
        ar & num_subnodes_;
        ar & children_;
        ar & parent_;
    }
}
