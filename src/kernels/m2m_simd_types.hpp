#include "taylor.hpp"

#include <Vc/Vc>


#if defined(Vc_HAVE_AVX512F)
using m2m_vector = Vc::datapar<double, Vc::datapar_abi::avx512>;
using m2m_int_vector = typename Vc::datapar<int32_t, Vc::datapar_abi::fixed_size<8>>;
// using m2m_int_vector = Vc::datapar<int64_t, Vc::datapar_abi::avx512>;
#elif defined(Vc_HAVE_AVX)    // assumes avx 2
using m2m_vector = typename Vc::datapar<double, Vc::datapar_abi::avx>;
using m2m_int_vector = typename Vc::datapar<int32_t, Vc::datapar_abi::fixed_size<4>>;
// using m2m_int_vector = typename Vc::datapar<int64_t, Vc::datapar_abi::avx>;
#else // falling back to fixed_size types
using m2m_vector = typename Vc::datapar<double, Vc::datapar_abi::fixed_size<8>>;
using m2m_int_vector = typename Vc::datapar<int64_t, Vc::datapar_abi::fixed_size<8>>;
#endif

using multipole_v = taylor<4, m2m_vector>;
using expansion_v = taylor<4, m2m_vector>;
