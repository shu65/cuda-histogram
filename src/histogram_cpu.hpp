#pragma once

#include <stdint.h>

void HistogramCPU(const int *data, const uint32_t n, const uint32_t n_bins, uint32_t *bin_counts);