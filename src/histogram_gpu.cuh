#pragma once

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

void PrepareHistogramGPUDefault(const cudaStream_t &stream);
void HistogramGPUv1(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *bin_counts);

void PrepareHistogramGPUv2(const cudaStream_t &stream, uint32_t *bin_counts);
void HistogramGPUv2(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *bin_counts);

void HistogramGPUv3(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *bin_counts);
void HistogramGPUv4(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *tmp_bin_counts, uint32_t *bin_counts);