#include <iostream>
#include <random>
#include <chrono>
#include <stdint.h>
#include <vector>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "cuda_helper.hpp"
#include "histogram_cpu.hpp"
#include "histogram_gpu.cuh"

using namespace std;

template <typename T>
bool EqualAarrays(const vector<T> &a1, const vector<T> &a2)
{
  size_t n = a1.size();
  if (n != a2.size())
  {
    return false;
  }
  for (size_t i = 0; i < n; ++i)
  {
    // cout << "i:" << i << " a1[i]" << a1[i] << " a2[i]" << a2[i] << endl;
    if (a1[i] != a2[i])
    {
      cout << "invalid i is " << i << " a1[i]" << a1[i] << " a2[i]" << a2[i] << endl;
      return false;
    }
  }
  return true;
}

void RunFunc(
    int func_id,
    const int *h_data,
    const int *d_data,
    const uint32_t n,
    const uint32_t n_bins,
    const cudaStream_t &stream,
    uint32_t *d_tmp_bin_counts,
    uint32_t *h_bin_counts,
    uint32_t *d_bin_counts)
{
  switch (func_id)
  {
  case 0:
    HistogramCPU(h_data, n, n_bins, h_bin_counts);
    break;
  case 1:
    HistogramGPUv1(d_data, n, n_bins, stream, d_bin_counts);
    break;
  case 2:
    HistogramGPUv2(d_data, n, n_bins, stream, d_bin_counts);
    break;
  case 3:
    HistogramGPUv3(d_data, n, n_bins, stream, d_tmp_bin_counts, d_bin_counts);
    break;
  default:
    assert(false);
  }
}

void Benchmark(const int max_func_id, const uint32_t n_bins, const bool using_l2_persisting_accesses)
{
  cout << "n bins:" << n_bins << endl;
  cout << "using_l2_persisting_accesses:" << using_l2_persisting_accesses << endl;
  const uint32_t log_n = 28;
  const uint32_t n = 1 << log_n;

  const uint32_t n_trials = 10;
  vector<int> h_data(n);
  vector<uint32_t> h_expected_bin_counts(n_bins);
  vector<uint32_t> h_actual_bin_counts(n_bins);
  std::mt19937 random_generator(42);
  std::uniform_int_distribution<> dist(0, n_bins - 1);

  for (uint32_t i = 0; i < n; ++i)
  {
    h_data[i] = dist(random_generator);
  }
  HistogramCPU(h_data.data(), n, n_bins, h_expected_bin_counts.data());

  cudaStream_t stream;
  int *d_data = nullptr;
  uint32_t *d_tmp_bin_counts = nullptr;
  uint32_t *d_actual_bin_counts = nullptr;

  CheckCudaErrors(cudaStreamCreate(&stream));

  CheckCudaErrors(cudaMalloc(&d_data, sizeof(int) * n));
  CheckCudaErrors(cudaMalloc(&d_tmp_bin_counts, sizeof(uint32_t) * n_bins * kMaxBlocks));
  CheckCudaErrors(cudaMalloc(&d_actual_bin_counts, sizeof(uint32_t) * n_bins));

  CheckCudaErrors(cudaMemcpy(d_data, h_data.data(), sizeof(int) * n, cudaMemcpyDefault));
  if (using_l2_persisting_accesses)
  {
    SetHistogramGPUL2PersistingAccesses(stream, n_bins, d_actual_bin_counts);
  }
  else
  {
    SetHistogramGPUL2Default(stream);
  }
  cudaDeviceSynchronize();
  for (int func_id = 0; func_id < max_func_id; ++func_id)
  {
    cudaStreamSynchronize(stream);
    // warm up
    for (uint32_t i = 0; i < 5; ++i)
    {
      RunFunc(
          func_id,
          h_data.data(),
          d_data,
          n,
          n_bins,
          stream,
          d_tmp_bin_counts,
          h_actual_bin_counts.data(),
          d_actual_bin_counts);
    }
    cudaStreamSynchronize(stream);
    chrono::system_clock::time_point start = chrono::system_clock::now();
    for (uint32_t i = 0; i < n_trials; ++i)
    {
      RunFunc(
          func_id,
          h_data.data(),
          d_data,
          n,
          n_bins,
          stream,
          d_tmp_bin_counts,
          h_actual_bin_counts.data(),
          d_actual_bin_counts);
    }
    cudaStreamSynchronize(stream);
    chrono::system_clock::time_point end = chrono::system_clock::now();
    double elapsed_time = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count()) / n_trials * 1e-6;
    cout << "func_id: " << func_id << ", array_size: " << n << ", time: " << elapsed_time << " sec." << endl;
    if (func_id > 0)
    {
      CheckCudaErrors(cudaMemcpy(h_actual_bin_counts.data(), d_actual_bin_counts, sizeof(uint32_t) * n_bins, cudaMemcpyDefault));
    }
    assert(EqualAarrays(h_expected_bin_counts, h_actual_bin_counts));
  }

  CheckCudaErrors(cudaStreamDestroy(stream));
  // free device memory
  CheckCudaErrors(cudaFree(d_data));
  d_data = nullptr;
  CheckCudaErrors(cudaFree(d_tmp_bin_counts));
  d_tmp_bin_counts = nullptr;
  CheckCudaErrors(cudaFree(d_actual_bin_counts));
  d_actual_bin_counts = nullptr;
}

int main(int argc, char **argv)
{
  Benchmark(4, 256, false);
  Benchmark(4, 256, true);
  Benchmark(1, 5 * (1 << 20), false);
  Benchmark(1, 5 * (1 << 20), true);
  cout << "complated!" << endl;
}