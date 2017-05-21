#include "geometry.hpp"
#include "interaction_types.hpp"
#include "interactions_iterators.hpp"

extern std::vector<interaction_type> ilist_debugging;

extern std::vector<interaction_type> ilist_n;
extern std::array<std::vector<interaction_type>, geo::direction::count()> ilist_n0_bnd;

void compare_interaction_lists() {
    uint64_t bnd_sum = std::accumulate(ilist_n0_bnd.begin(), ilist_n0_bnd.end(), 0.0,
        [](int sum, std::vector<interaction_type> e) { return sum + e.size(); });
    std::cout << "ilist_debugging.size(): " << ilist_debugging.size() << std::endl;
    std::cout << "ilist_n.size(): " << ilist_n.size() << std::endl;
    std::cout << "ilist_n0_bnd.size(): " << ilist_n0_bnd.size() << std::endl;
    std::cout << "bnd_sum: " << bnd_sum << std::endl;
    if (ilist_debugging.size() != ilist_n.size() + bnd_sum) {
        std::cout << "error: sizes of interaction lists do not match" << std::endl;
    }

    std::cout << "check which interactions of new kernel cannot be found in either old list"
              << std::endl;
    for (interaction_type& mine : ilist_debugging) {
        octotiger::fmm::multiindex mine_first_padded =
            octotiger::fmm::flat_index_to_multiindex_padded(mine.first);
        octotiger::fmm::multiindex mine_second_padded =
            octotiger::fmm::flat_index_to_multiindex_padded(mine.second);

        octotiger::fmm::multiindex mine_first(
            mine_first_padded.x - 8, mine_first_padded.y - 8, mine_first_padded.z - 8);
        octotiger::fmm::multiindex mine_second(
            mine_second_padded.x - 8, mine_second_padded.y - 8, mine_second_padded.z - 8);
        size_t mine_first_flat = octotiger::fmm::to_inner_flat_index_not_padded(mine_first);
        const integer mine_second_flat = gindex(
            (mine_second.x + INX) % INX, (mine_second.y + INX) % INX, (mine_second.z + INX) % INX);
        // size_t mine_second_flat = octotiger::fmm::to_inner_flat_index_not_padded(mine_second);

        bool found = false;
        for (interaction_type& ref : ilist_n) {
            if (mine_first_flat == ref.first && mine_second_flat == ref.second) {
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
                if (mine_first_flat == ref.first && mine_second_flat == ref.second) {
                    found = true;
                    std::cout << "found boundary!" << std::endl;
                    break;
                }
            }
        }
        if (found) {
            continue;
        }

        octotiger::fmm::multiindex i = octotiger::fmm::flat_index_to_multiindex_padded(mine.first);
        octotiger::fmm::multiindex partner =
            octotiger::fmm::flat_index_to_multiindex_padded(mine.second);
        octotiger::fmm::multiindex diff(partner.x - i.x, partner.y - i.y, partner.z - i.z);
        std::cout << "interaction not found i_f: " << mine.first << " partner_f: " << mine.second
                  << " i: " << i << " partner: " << partner << " diff: " << diff << std::endl;
    }

    std::cout << "check whether old inner list -> in new list" << std::endl;

    uint64_t no_found = 0;
    uint64_t no_not_found = 0;

    for (interaction_type& mine : ilist_n) {
        octotiger::fmm::multiindex i =
            octotiger::fmm::flat_index_to_multiindex_not_padded(mine.first);
        octotiger::fmm::multiindex partner =
            octotiger::fmm::flat_index_to_multiindex_not_padded(mine.second);
        octotiger::fmm::multiindex diff(partner.x - i.x, partner.y - i.y, partner.z - i.z);

        bool found = false;
        for (interaction_type& ref : ilist_debugging) {
            octotiger::fmm::multiindex ref_first_padded =
                octotiger::fmm::flat_index_to_multiindex_padded(ref.first);
            octotiger::fmm::multiindex ref_second_padded =
                octotiger::fmm::flat_index_to_multiindex_padded(ref.second);

            octotiger::fmm::multiindex ref_first(
                ref_first_padded.x - 8, ref_first_padded.y - 8, ref_first_padded.z - 8);
            octotiger::fmm::multiindex ref_second(
                ref_second_padded.x - 8, ref_second_padded.y - 8, ref_second_padded.z - 8);
            size_t ref_first_flat = octotiger::fmm::to_inner_flat_index_not_padded(ref_first);
            size_t ref_second_flat = octotiger::fmm::to_inner_flat_index_not_padded(ref_second);

            if (mine.first == ref_first_flat && mine.second == ref_second_flat) {
                found = true;
                octotiger::fmm::multiindex partner_debug(
                    i.x + ref.x[0], i.y + ref.x[1], i.z + ref.x[2]);
                octotiger::fmm::multiindex diff_debug(ref.x[0], ref.x[1], ref.x[2]);
                std::cout << "found!"
                          << " i: " << i << " partner: " << partner << " diff: " << diff
                          << " diff_len: " << diff.length() << std::endl;
                // std::cout << "diff_debug: " << diff_debug << std::endl;
                no_found += 1;
                break;
            }
        }
        if (found) {
            continue;
        }

        std::cout << "old inner interaction not found o_f: " << mine.first
                  << " o_partner_f: " << mine.second << " i: " << i << " partner: " << partner
                  << " diff: " << diff << " diff_len: " << diff.length() << std::endl;
        no_not_found += 1;
    }

    std::cout << "no_found: " << no_found << std::endl;
    std::cout << "no_not_found: " << no_not_found << std::endl;

    std::cout << "check whether old boundary list -> in new list" << std::endl;

    no_found = 0;
    no_not_found = 0;

    for (geo::direction& dir : geo::direction::full_set()) {
        for (interaction_type& mine : ilist_n0_bnd[dir]) {
            octotiger::fmm::multiindex i(
                mine.first_index[0], mine.first_index[1], mine.first_index[2]);
            octotiger::fmm::multiindex partner(
                mine.second_index[0], mine.second_index[1], mine.second_index[1]);
            octotiger::fmm::multiindex diff(partner.x - i.x, partner.y - i.y, partner.z - i.z);

            bool found = false;
            for (interaction_type& ref : ilist_debugging) {
                octotiger::fmm::multiindex ref_first_padded =
                    octotiger::fmm::flat_index_to_multiindex_padded(ref.first);
                octotiger::fmm::multiindex ref_second_padded =
                    octotiger::fmm::flat_index_to_multiindex_padded(ref.second);

                octotiger::fmm::multiindex ref_first(
                    ref_first_padded.x % 8, ref_first_padded.y % 8, ref_first_padded.z % 8);
                octotiger::fmm::multiindex ref_second(
                    ref_second_padded.x % 8, ref_second_padded.y % 8, ref_second_padded.z % 8);
                size_t ref_first_flat = octotiger::fmm::to_inner_flat_index_not_padded(ref_first);
                size_t ref_second_flat = octotiger::fmm::to_inner_flat_index_not_padded(ref_second);

                if (mine.first == ref_first_flat && mine.second == ref_second_flat) {
                    found = true;
                    octotiger::fmm::multiindex partner_debug(
                        i.x + ref.x[0], i.y + ref.x[1], i.z + ref.x[2]);
                    octotiger::fmm::multiindex diff_debug(ref.x[0], ref.x[1], ref.x[2]);
                    std::cout << "old found boundary!"
                              << " i: " << i << " partner: " << partner << " diff: " << diff
                              << " diff_len: " << diff.length() << std::endl;
                    // std::cout << "diff_debug: " << diff_debug << std::endl;
                    no_found += 1;
                    break;
                }
            }
            if (found) {
                continue;
            }

            std::cout << "old boundary interaction not found o_f: " << mine.first
                      << " o_partner_f: " << mine.second << " i: " << i << " partner: " << partner
                      << " diff: " << diff << " diff_len: " << diff.length() << std::endl;
            no_not_found += 1;
        }
    }

    std::cout << "no_found: " << no_found << std::endl;
    std::cout << "no_not_found: " << no_not_found << std::endl;
}
