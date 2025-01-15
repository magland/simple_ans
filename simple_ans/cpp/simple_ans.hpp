#pragma once

#include <vector>
#include <stdexcept>
#include <cstdint>

namespace simple_ans {

struct EncodedData {
    uint32_t state;
    std::vector<uint64_t> bitstream; // Each uint64_t contains 64 bits, with padding in last word if needed
    size_t num_bits; // Actual number of bits used (may be less than bitstream.size() * 64)
};

// Helper function to verify if a number is a power of 2
inline bool is_power_of_2(uint32_t x) {
    return x && !(x & (x - 1));
}

// Performs streaming Asymmetric Numeral Systems (ANS) encoding of a signal
// signal_size: number of elements in signal array
// num_symbols: length of symbol_counts and symbol_values arrays
// Returns: EncodedData containing the encoded state and bitstream
EncodedData encode(const int32_t* signal, size_t signal_size,
                  const uint32_t* symbol_counts, const int32_t* symbol_values, size_t num_symbols);

// Performs streaming Asymmetric Numeral Systems (ANS) decoding of an encoded signal
// output: pre-allocated buffer to store decoded signal (must be size n)
// n: number of elements to decode
// num_symbols: length of symbol_counts and symbol_values arrays
void decode(int32_t* output, size_t n,
           uint32_t state, const uint64_t* bitstream, size_t num_bits,
           const uint32_t* symbol_counts, const int32_t* symbol_values, size_t num_symbols);

// Convert real-valued proportions into integer counts summing to L
// proportions: array of real-valued proportions
// num_proportions: length of proportions array
// counts_out: pre-allocated buffer to store resulting counts (must be size num_proportions)
void choose_symbol_counts(uint32_t* counts_out, const double* proportions, size_t num_proportions, uint32_t L);

// Test function that adds 1 to each element of an array - for benchmarking data transfer
// output: pre-allocated buffer to store result (must be same size as input)
// size: number of elements in input/output arrays
void add_one_test(int32_t* output, const int32_t* input, size_t size);

} // namespace simple_ans
