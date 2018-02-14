#pragma once

#ifndef NVCC
#include "kernel_simd_types.hpp"
#endif

// #include "interaction_constants.hpp"

namespace octotiger {
namespace fmm {

    // component type has to be indexable, and has to have a size() operator
    template <typename component_type, size_t num_components, size_t entries, size_t padding>
    class struct_of_array_data
    {
    private:
        // data in SoA form
        static constexpr size_t padded_entries_per_component = entries + padding;

        component_type* const data;

    public:
        template <size_t component_access>
        inline component_type* pointer(const size_t flat_index) const {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            // should result in single move instruction, indirect addressing: reg + reg +
            // constant
            return data + flat_index + component_array_offset;
        }

#ifndef NVCC
        // careful, this returns a copy!
        template <size_t component_access>
        inline m2m_vector value(const size_t flat_index) const {
            return m2m_vector(
                this->pointer<component_access>(flat_index), Vc::flags::element_aligned);
        }
#endif

        // template <size_t component_access>
        // inline component_type& reference(const size_t flat_index) const {
        //     return *this->pointer<component_access>(flat_index);
        // }
        template <typename AoS_temp_type>
        void set_AoS_value(AoS_temp_type&& value, size_t flatindex) {
            for (size_t component = 0; component < num_components; component++) {
                data[component * padded_entries_per_component + flatindex] = value[component];
            }
        }
        template <typename AoS_temp_type>
        inline void set_value(AoS_temp_type&& value, size_t flatindex) {
            data[flatindex] = value;
            for (size_t component = 1; component < num_components; component++) {
                data[component * padded_entries_per_component + flatindex] = 0.0;
            }
        }

        struct_of_array_data(void)
          : data(new component_type[num_components * padded_entries_per_component]) {
            for (auto i = 0; i < num_components * padded_entries_per_component; ++i) {
                data[i] = 0.0;
            }
        }

        struct_of_array_data(const size_t entries_per_component)
          : data(new component_type[num_components * padded_entries_per_component]) {}

        ~struct_of_array_data() {
            delete[] data;
        }

        struct_of_array_data(const struct_of_array_data& other) = delete;

        struct_of_array_data(const struct_of_array_data&& other) = delete;

        struct_of_array_data& operator=(const struct_of_array_data& other) = delete;

#ifndef NVCC
        // write back into non-SoA style array
        template <typename AoS_type>
        void to_non_SoA(std::vector<AoS_type>& org) {
            // constexpr size_t padded_entries_per_component = entries + padding;
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    org[entry][component] = data[component * padded_entries_per_component + entry];
                }
            }
        }
        template <typename AoS_type>
        void add_to_non_SoA(std::vector<AoS_type>& org) {
            // constexpr size_t padded_entries_per_component = entries + padding;
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    org[entry][component] += data[component * padded_entries_per_component + entry];
                }
            }
        }
#endif

        inline double* get_underlying_pointer() {
            return data;
        }
        inline size_t size() {
            return num_components * padded_entries_per_component;
        }

        inline size_t get_size_bytes() {
            return num_components * padded_entries_per_component * sizeof(component_type);
        }

        component_type& at(size_t component_access, size_t flat_index) {
            size_t component_array_offset = component_access * padded_entries_per_component;
            return *(data + flat_index + component_array_offset);
        }

#ifndef NVCC
        template <typename AoS_type>
        static struct_of_array_data<component_type, num_components, entries, padding> from_vector(
            const std::vector<AoS_type>& org) {
            struct_of_array_data r;
            component_type* data = r.get_underlying_pointer();
            for (size_t component = 0; component < num_components; component++) {
                for (size_t entry = 0; entry < org.size(); entry++) {
                    data[component * padded_entries_per_component + entry] = org[entry][component];
                }
            }
            return r;
        }
#endif
    };
}    // namespace fmm
}    // namespace octotiger
