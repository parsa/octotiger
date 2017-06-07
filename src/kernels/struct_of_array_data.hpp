#pragma once

namespace octotiger {
namespace fmm {

    template <typename AoS_type, typename component_type, size_t num_components>
    class struct_of_array_data;

    template <typename AoS_type, typename component_type, size_t num_components>
    class struct_of_array_view
    {
    protected:
        struct_of_array_data<AoS_type, component_type, num_components>& data;
        size_t flat_index;

    public:
        struct_of_array_view<AoS_type, component_type, num_components>(
            struct_of_array_data<AoS_type, component_type, num_components>& data, size_t flat_index)
          : data(data)
          , flat_index(flat_index) {}

        // returning pointer so that Vc can convert into simdarray
        // (notice: a reference would be silently broadcasted)
        component_type* component_pointer(size_t component_index) const {
            return data.access(component_index, flat_index);
        }
    };

    // component type has to be indexable, and has to have a size() operator
    template <typename AoS_type, typename component_type, size_t num_components>
    class struct_of_array_data
    {
    private:
        // data in SoA form
        std::vector<component_type> data;
        const size_t entries_per_component;

    public:
        struct_of_array_data(const std::vector<AoS_type>& org)
          : entries_per_component(org.size()) {
            data = std::vector<component_type>(num_components * org.size());
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    data[component * entries_per_component + entry] = org[entry][component];
                }
            }
        }

        inline component_type* access(const size_t component_index, const size_t flat_entry_index) {
            return &data[component_index * entries_per_component + flat_entry_index];
        }

        struct_of_array_view<AoS_type, component_type, num_components> get_view(size_t flat_index) {
            return struct_of_array_view<AoS_type, component_type, num_components>(
                *this, flat_index);
        }

        // write back into non-SoA style array
        void to_non_SoA(std::vector<AoS_type> &org) {
        for (size_t component = 0; component < num_components; component++) {
            for (size_t entry = 0; entry < org.size(); entry++) {
                org[entry][component] = data[component * entries_per_component + entry];
            }
        }
        }
    };

}    // namespace fmm
}    // namespace octotiger
