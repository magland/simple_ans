#include "simple_ans.hpp"
#include <iostream>
#include <unordered_map>

// From: https://graphallthethings.com/posts/streaming-ans-explained/

namespace simple_ans
{

EncodedData encode(const int32_t* signal,
                   size_t signal_size,
                   const uint32_t* symbol_counts,
                   const int32_t* symbol_values,
                   size_t num_symbols)
{
    // Create mapping from values to indices
    std::unordered_map<int32_t, size_t> symbol_index_for_value;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        symbol_index_for_value[symbol_values[i]] = i;
    }

    // Calculate L (sum of symbol counts) and verify it's a power of 2
    uint32_t L = 0;
    for (size_t i = 0; i < num_symbols; ++i)
    {
        L += symbol_counts[i];
    }
    std::cout << "L = " << L << std::endl;

    if (!is_power_of_2(L))
    {
        throw std::invalid_argument("L must be a power of 2");
    }
    std::cout << "Symbol counts: ";
    for (size_t i = 0; i < num_symbols; ++i)
    {
        std::cout << symbol_counts[i] << " ";
    }
    std::cout << std::endl;

    // Pre-compute cumulative sums for efficiency
    std::vector<uint32_t> cumsum(num_symbols + 1, 0);
    for (size_t i = 0; i < num_symbols; ++i)
    {
        cumsum[i + 1] = cumsum[i] + symbol_counts[i];
    }

    // Initialize state and packed bitstream
    uint32_t state = L;
    std::vector<uint64_t> bitstream;
    bitstream.reserve(signal_size / 32);  // Reserve conservative space (64-bit words)
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
            size_t word_idx = num_bits / 64;
            size_t bit_idx = num_bits % 64;
            if (word_idx >= bitstream.size())
            {
                bitstream.push_back(0);
            }
            if (state_normalized & 1)
            {
                bitstream[word_idx] |= (1ull << bit_idx);
            }
            num_bits++;
            state_normalized >>= 1;
        }

        // Update state
        state = L + cumsum[s] + state_normalized - L_s;

        if (state < L || state >= 2 * L)
        {
            throw std::runtime_error("Invalid state during encoding");
        }
    }

    return {state, std::move(bitstream), num_bits};
}

void decode(int32_t* output,
            size_t n,
            uint32_t state,
            const uint64_t* bitstream,
            size_t num_bits,
            const uint32_t* symbol_counts,
            const int32_t* symbol_values,
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
    std::vector<uint8_t> symbol_lookup(L);
    std::vector<uint32_t> cumsum(num_symbols + 1, 0);

    // Build cumulative sums and lookup table
    for (size_t i = 0; i < num_symbols; ++i)
    {
        cumsum[i + 1] = cumsum[i] + symbol_counts[i];
        for (uint32_t j = cumsum[i]; j < cumsum[i + 1]; ++j)
        {
            symbol_lookup[j] = static_cast<uint8_t>(i);
        }
    }

    // Prepare bit reading
    int64_t bit_pos = num_bits - 1;
    const uint32_t L_mask = L - 1;  // For fast modulo since L is power of 2

    // Decode symbols in reverse order with optimized operations
    for (size_t i = 0; i < n; ++i)
    {
        // Find symbol using lookup table instead of binary search
        uint32_t remainder = state & L_mask;  // Fast modulo for power of 2
        uint8_t s = symbol_lookup[remainder];

        // Calculate normalized state
        uint32_t state_normalized = symbol_counts[s] + state - L - cumsum[s];

        state = state_normalized;

        while (state < L)
        {
            if (bit_pos < 0)
            {
                throw std::runtime_error("Bitstream exhausted");
            }
            uint32_t word_idx = bit_pos >> 6;  // Divide by 64
            uint32_t bit_idx = bit_pos & 63;   // Modulo 64
            state = (state << 1) | ((bitstream[word_idx] >> bit_idx) & 1);
            --bit_pos;
        }

        output[n - 1 - i] = symbol_values[s];
    }
}

void choose_symbol_counts(uint32_t* counts_out,
                          const double* proportions,
                          size_t num_proportions,
                          uint32_t L)
{
    if (num_proportions > L)
    {
        throw std::invalid_argument("More proportions than items to distribute");
    }

    // normalize the proportions to sum to 1
    double sum = 0;
    for (size_t i = 0; i < num_proportions; ++i)
    {
        sum += proportions[i];
    }
    std::vector<double> normalized_props(num_proportions);
    for (size_t i = 0; i < num_proportions; ++i)
    {
        normalized_props[i] = proportions[i] / sum;
    }

    // first give everyone one to start
    for (size_t i = 0; i < num_proportions; ++i)
    {
        counts_out[i] = 1;
    }
    uint32_t remaining = L - num_proportions;

    // real-valued target counts
    std::vector<double> target_counts(num_proportions);
    for (size_t i = 0; i < num_proportions; ++i)
    {
        target_counts[i] = normalized_props[i] * L;
    }

    while (remaining > 0)
    {
        std::vector<double> residuals(num_proportions);
        std::vector<int32_t> residuals_int_part(num_proportions);
        std::vector<double> residuals_frac_part(num_proportions);

        for (size_t i = 0; i < num_proportions; ++i)
        {
            residuals[i] = target_counts[i] - counts_out[i];
            residuals_int_part[i] = static_cast<int32_t>(residuals[i]);
            residuals_frac_part[i] = residuals[i] - residuals_int_part[i];
        }

        // Check if any integer parts are positive
        bool has_positive_int = false;
        for (const auto& r : residuals_int_part)
        {
            if (r > 0)
            {
                has_positive_int = true;
                break;
            }
        }

        if (has_positive_int)
        {
            // Distribute based on integer parts
            for (size_t i = 0; i < num_proportions; ++i)
            {
                if (residuals_int_part[i] > 0)
                {
                    uint32_t to_add =
                        std::min(static_cast<uint32_t>(residuals_int_part[i]), remaining);
                    counts_out[i] += to_add;
                    remaining -= to_add;
                    if (remaining == 0)
                        break;
                }
            }
        }
        else
        {
            // Find index with largest fractional part
            size_t max_idx = 0;
            double max_frac = residuals_frac_part[0];
            for (size_t i = 1; i < residuals_frac_part.size(); ++i)
            {
                if (residuals_frac_part[i] > max_frac)
                {
                    max_frac = residuals_frac_part[i];
                    max_idx = i;
                }
            }
            counts_out[max_idx]++;
            remaining--;
        }
    }
}

void add_one_test(int32_t* output, const int32_t* input, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        output[i] = input[i] + 1;
    }
}

}  // namespace simple_ans
