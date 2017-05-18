#include "geometry.hpp"
#include "interaction_types.hpp"
#include "interactions_iterators.hpp"

extern std::vector<interaction_type> ilist_debugging;

extern std::vector<interaction_type> ilist_n;
extern std::array<std::vector<interaction_type>, geo::direction::count()> ilist_n0_bnd;

void compare_interaction_lists() {
    if (ilist_debugging.size() != ilist_n.size() + ilist_n0_bnd.size()) {
        std::cout << "error: sizes of interaction lists do not match" << std::endl;
    }

    std::cout << "check which interactions of new kernel cannot be found in either old list"
              << std::endl;
    for (interaction_type& mine : ilist_debugging) {
        bool found = false;
        for (interaction_type& ref : ilist_n) {
            if (mine.first == ref.first && mine.second == ref.second) {
                found = true;
                std::cout << "found inner!" << std::endl;
                break;
            }
        }
        if (found) {
            continue;
        }
        for (geo::direction& dir : geo::direction::full_set()) {
            for (interaction_type& ref : ilist_n0_bnd[dir]) {
                if (mine.first == ref.first && mine.second == ref.second) {
                    found = true;
                    std::cout << "found boundary!" << std::endl;
                    break;
                }
            }
        }
        if (found) {
            continue;
        }

        octotiger::fmm::multiindex i = octotiger::fmm::inner_flat_index_to_multiindex(mine.first);
        octotiger::fmm::multiindex partner =
            octotiger::fmm::inner_flat_index_to_multiindex(mine.second);
        octotiger::fmm::multiindex diff(i.x - partner.x, i.y - partner.y, i.z - partner.z);
        std::cout << "interaction not found i_f: " << mine.first << " partner_f: " << mine.second
                  << " i: " << i << " partner: " << partner << " diff: " << diff << std::endl;
    }

    // std::cout << "check whether old inner list -> in new list" << std::endl;
    // for (interaction_type& mine : ilist_n) {
    //     bool found = false;
    //     for (interaction_type& ref : ilist_debugging) {
    //         if (mine.first == ref.first && mine.second == ref.second) {
    //             found = true;
    //             std::cout << "found!" << std::endl;
    //             break;
    //         }
    //     }
    //     if (found) {
    //         continue;
    //     }

    //     std::cout << "old inner interaction not found o_f: " << mine.first
    //               << " o_partner_f: " << mine.second << std::endl;
    // }

    // std::cout << "check whether old boundary list -> in new list" << std::endl;
    // for (geo::direction& dir : geo::direction::full_set()) {
    //     for (interaction_type& mine : ilist_n0_bnd[dir]) {
    //         bool found = false;
    //         for (interaction_type& ref : ilist_debugging) {
    //             if (mine.first == ref.first && mine.second == ref.second) {
    //                 found = true;
    //                 std::cout << "found!" << std::endl;
    //                 break;
    //             }
    //         }
    //         if (found) {
    //             continue;
    //         }

    //         std::cout << "old boundary interaction not found o_f: " << mine.first
    //                   << " o_partner_f: " << mine.second << std::endl;
    //     }
    // }
}
