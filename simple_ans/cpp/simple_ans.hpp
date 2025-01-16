#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Template Implementation
////////////////////////////////////////////////////////////////////////////////

namespace simple_ans
{

template <typename T>
EncodedData encode_t(const T* signal,
                     size_t signal_size,
                     const uint32_t* symbol_counts,
                     const T* symbol_values,
                     size_t num_symbols)
{
    // Calculate L (sum of symbol counts) and verify it's a power of 2
    uint32_t L = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        L += symbol_counts[i];
    }

    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    // Pre-compute cumulative sums for efficiency
    std::vector<uint32_t> cumsum(num_symbols + 1, 0);
    for (size_t i = 0; i < num_symbols; ++i)
    {
        cumsum[i + 1] = cumsum[i] + symbol_counts[i];
    }

    // Create mapping from values to indices
    std::unordered_map<T, size_t> symbol_index_for_value;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        symbol_index_for_value[symbol_values[i]] = i;
    }

    // Initialize state and packed bitstream
    uint32_t state = L;
    // Preallocate bitstream - in worst case, each symbol needs log2(2L) bits
    // Convert to 64-bit words by dividing by 64 and rounding up
    size_t max_bits = signal_size * (32 - __builtin_clz(2 * L - 1));
    size_t num_words = (max_bits + 63) / 64;
    std::vector<uint64_t> bitstream(num_words, 0);
    size_t num_bits = 0;

    // Encode each symbol
    for (size_t i = 0; i < signal_size; ++i)
    {
        auto it = symbol_index_for_value.find(signal[i]);
        if (it == symbol_index_for_value.end())
        {
            throw std::invalid_argument("Signal value not found in symbol_values");
        }
        const auto s = it->second;
        uint32_t state_normalized = state;
        const uint32_t L_s = symbol_counts[s];

        // Normalize state
        // we need state_normalized to be in the range [L_s, 2*L_s)
        while (state_normalized >= 2 * L_s)
        {
            // Add bit to packed format
            size_t word_idx = num_bits >> 6;  // Divide by 64
            size_t bit_idx = num_bits & 63;   // Modulo 64
            bitstream[word_idx] |= static_cast<uint64_t>(state_normalized & 1) << bit_idx;
            num_bits++;
            state_normalized >>= 1;
        }

        // Update state
        state = L + cumsum[s] + state_normalized - L_s;
    }

    // Truncate bitstream to actual size used
    size_t final_words = (num_bits + 63) / 64;
    bitstream.resize(final_words);
    return {state, std::move(bitstream), num_bits};
}

template <typename T>
void decode_t(T* output,
              size_t n,
              uint32_t state,
              const uint64_t* bitstream,
              size_t num_bits,
              const uint32_t* symbol_counts,
              const T* symbol_values,
              size_t num_symbols)
{
    // Calculate L and verify it's a power of 2
    uint32_t L = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        L += symbol_counts[i];
    }

    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }

    if (state < L || state >= 2 * L)
    {
        throw std::invalid_argument("Initial state is invalid");
    }

    // Pre-compute lookup table for symbol finding
    std::vector<uint32_t> symbol_lookup(L);
    std::vector<uint32_t> cumsum(num_symbols + 1, 0);

    // Build cumulative sums and lookup table
    for (size_t i = 0; i < num_symbols; ++i)
    {
        cumsum[i + 1] = cumsum[i] + symbol_counts[i];
        for (uint32_t j = cumsum[i]; j < cumsum[i + 1]; ++j)
        {
            symbol_lookup[j] = static_cast<uint32_t>(i);
        }
    }

    // Prepare bit reading
    int64_t bit_pos = num_bits - 1;
    const uint32_t L_mask = L - 1;  // For fast modulo since L is power of 2

    // Decode symbols in reverse order
    for (size_t i = 0; i < n; ++i)
    {
        // Find symbol using lookup table instead of binary search
        uint32_t remainder = state & L_mask;  // Fast modulo for power of 2
        uint32_t s = symbol_lookup[remainder];

        // Calculate normalized state which will be in the range [L_s, 2*L_s)
        uint32_t L_s = symbol_counts[s];
        state = L_s + state - L - cumsum[s];

        // read from the bit stream and get us back to the range [L, 2L)
        // determine how many bits need to be read
        uint32_t num_bits_needed = 0;
        while ((state << num_bits_needed) < L)
        {
            num_bits_needed++;
        }
        if (num_bits_needed > 0) {
            if ((bit_pos & 63) >= (num_bits_needed - 1)) {
                // in this case we can grab all the bits we need at once from the current word
                // note: it wasn't clear that this trick was making a difference in performance
                uint32_t word_idx = bit_pos >> 6;  // Divide by 64
                uint32_t bit_idx = bit_pos & 63;   // Modulo 64
                // get bits from bit_idx - num_bits_needed + 1 to bit_idx
                uint32_t bits = static_cast<uint32_t>((bitstream[word_idx] >> (bit_idx - num_bits_needed + 1)) & ((1 << num_bits_needed) - 1));
                state = (state << num_bits_needed) | bits;
                bit_pos -= num_bits_needed;
            }
            else {
                // this is possibly the slower case, but should be less common
                for (uint32_t j = 0; j < num_bits_needed; ++j)
                {
                    uint32_t word_idx = bit_pos >> 6;  // Divide by 64
                    uint32_t bit_idx = bit_pos & 63;   // Modulo 64
                    state = (state << 1) | ((bitstream[word_idx] >> bit_idx) & 1);
                    bit_pos--;
                }
            }
        }

        output[n - 1 - i] = symbol_values[s];
    }
}

}  // namespace simple_ans
