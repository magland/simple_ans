#include "simple_ans.hpp"

namespace simple_ans {

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
        std::vector<uint32_t> residuals_int_part(num_proportions);
        std::vector<double> residuals_frac_part(num_proportions);

        for (size_t i = 0; i < num_proportions; ++i)
        {
            residuals[i] = target_counts[i] - counts_out[i];
            // Ensure we don't convert negative values to uint32_t
            residuals_int_part[i] = residuals[i] > 0 ? static_cast<uint32_t>(residuals[i]) : 0;
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

} // namespace simple_ans
