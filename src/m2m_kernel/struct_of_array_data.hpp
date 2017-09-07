#pragma once

#ifndef NVCC
#include "m2m_simd_types.hpp"
#endif

// #include "interaction_constants.hpp"

#include <vector>

namespace octotiger {
namespace fmm {

    // component type has to be indexable, and has to have a size() operator
    template <typename component_type, size_t num_components, size_t entries, size_t padding>
    class struct_of_array_data
    {
    private:
        // data in SoA form
        static constexpr size_t padded_entries_per_component = entries + padding;

        component_type* data;

        // struct_of_array_data()
        //   : data(nullptr) {}

    public:
        template <size_t component_access>
        inline component_type* pointer(const size_t flat_index) const {
            constexpr size_t component_array_offset =
                component_access * padded_entries_per_component;
            // should result in single move instruction, indirect addressing: reg + reg + constant
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

        // constructor that works on preallocated and initialized data
        struct_of_array_data(component_type* preallocated_data)
          : data(preallocated_data) {}

        struct_of_array_data()
          : data(new component_type[num_components * padded_entries_per_component]) {}

        ~struct_of_array_data() {
            delete[] data;
        }

        struct_of_array_data(const struct_of_array_data& other) = delete;

        struct_of_array_data(struct_of_array_data&& other)
          : data(other.data) {
            other.data = nullptr;
        }

        struct_of_array_data& operator=(const struct_of_array_data& other) = delete;

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
    };

}    // namespace fmm
}    // namespace octotiger
