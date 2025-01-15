#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace simple_ans
{

struct EncodedData
{
    uint32_t state;
    std::vector<uint64_t>
        bitstream;    // Each uint64_t contains 64 bits, with padding in last word if needed
    size_t num_bits;  // Actual number of bits used (may be less than bitstream.size() * 64)
};

// Helper function to verify if a number is a power of 2
inline bool is_power_of_2(uint32_t x)
{
    return x && !(x & (x - 1));
}

// Template version for different integer types
template <typename T>
EncodedData encode_t(const T* signal,
                     size_t signal_size,
                     const uint32_t* symbol_counts,
                     const T* symbol_values,
                     size_t num_symbols);

template <typename T>
void decode_t(T* output,
              size_t n,
              uint32_t state,
              const uint64_t* bitstream,
              size_t num_bits,
              const uint32_t* symbol_counts,
              const T* symbol_values,
              size_t num_symbols);

// Legacy int32_t versions for backward compatibility
inline EncodedData encode(const int32_t* signal,
                          size_t signal_size,
                          const uint32_t* symbol_counts,
                          const int32_t* symbol_values,
                          size_t num_symbols)
{
    return encode_t(signal, signal_size, symbol_counts, symbol_values, num_symbols);
}

inline void decode(int32_t* output,
                   size_t n,
                   uint32_t state,
                   const uint64_t* bitstream,
                   size_t num_bits,
                   const uint32_t* symbol_counts,
                   const int32_t* symbol_values,
                   size_t num_symbols)
{
    decode_t(output, n, state, bitstream, num_bits, symbol_counts, symbol_values, num_symbols);
}

// Convert real-valued proportions into integer counts summing to L
// proportions: array of real-valued proportions
// num_proportions: length of proportions array
// counts_out: pre-allocated buffer to store resulting counts (must be size num_proportions)
void choose_symbol_counts(uint32_t* counts_out,
                          const double* proportions,
                          size_t num_proportions,
                          uint32_t L);

}  // namespace simple_ans

#include "simple_ans.tpp"
