#include <stdint.h>

#include "common.hpp"

void HistogramCPU(const int *data, const uint32_t n, uint32_t *bin_counts) {
  for (uint32_t i = 0; i < kNBins; ++i) {
    bin_counts[i] = 0;
  }

  for (uint32_t i = 0; i < n; ++i) {
    const int bin_i = data[i];
    ++bin_counts[bin_i];
  }
}