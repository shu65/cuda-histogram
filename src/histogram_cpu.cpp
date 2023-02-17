#include <stdint.h>

void HistogramCPU(const int *data, const uint32_t n, const uint32_t n_bins, uint32_t *bin_counts)
{
  for (uint32_t i = 0; i < n_bins; ++i)
  {
    bin_counts[i] = 0;
  }

  for (uint32_t i = 0; i < n; ++i)
  {
    const int bin_i = data[i];
    ++bin_counts[bin_i];
  }
}