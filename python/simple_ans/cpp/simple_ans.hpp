#pragma once

#include <vector>
#include <stdexcept>
#include <cstdint>

namespace simple_ans {

struct EncodedData {
    uint32_t state;
    std::vector<uint8_t> bitstream; // Each byte contains 8 bits, with padding in last byte if needed
    size_t num_bits; // Actual number of bits used (may be less than bitstream.size() * 8)
};

// Performs streaming Asymmetric Numeral Systems (ANS) encoding of a signal
EncodedData encode(const std::vector<uint32_t>& signal, const std::vector<uint32_t>& symbol_counts);

// Performs streaming Asymmetric Numeral Systems (ANS) decoding of an encoded signal
std::vector<uint32_t> decode(uint32_t state, const std::vector<uint8_t>& bitstream, size_t num_bits,
                            const std::vector<uint32_t>& symbol_counts, size_t n);

// Helper function to verify if a number is a power of 2
inline bool is_power_of_2(uint32_t x) {
    return x && !(x & (x - 1));
}

// Convert real-valued proportions into integer counts summing to L
std::vector<uint32_t> choose_symbol_counts(const std::vector<double>& proportions, uint32_t L);

} // namespace simple_ans
