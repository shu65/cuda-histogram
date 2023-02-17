#pragma once

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

void SetHistogramGPUL2Default(const cudaStream_t &stream);
void SetHistogramGPUL2PersistingAccesses(const cudaStream_t &stream, const uint32_t n_bins, uint32_t *bin_counts);

void HistogramGPUv1(const int *data, const uint32_t n, const uint32_t n_bins, const cudaStream_t &stream, uint32_t *bin_counts);
void HistogramGPUv2(const int *data, const uint32_t n, const uint32_t n_bins, const cudaStream_t &stream, uint32_t *bin_counts);
void HistogramGPUv3(const int *data, const uint32_t n, const uint32_t n_bins, const cudaStream_t &stream, uint32_t *tmp_bin_counts, uint32_t *bin_counts);