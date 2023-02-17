#include <iostream>

#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cuda_helper.hpp"
#include "common.hpp"



namespace cg = cooperative_groups;
using namespace std;


cudaAccessPolicyWindow GetDefalutAccessPolicyWindow() {
  cudaAccessPolicyWindow accessPolicyWindow = {0};
  accessPolicyWindow.base_ptr = (void *)0;
  accessPolicyWindow.num_bytes = 0;
  accessPolicyWindow.hitRatio = 0.f;
  accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return accessPolicyWindow;
}

void PrepareHistogramGPUDefault(const cudaStream_t &stream) {
  cudaDeviceProp prop;
  CheckCudaErrors(cudaGetDeviceProperties(&prop, 0));
  cout << "l2CacheSize:" <<  prop.l2CacheSize << endl;
  cout << "persistingL2CacheMaxSize:" <<   prop.persistingL2CacheMaxSize << endl;
  
  CheckCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize));
  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow = GetDefalutAccessPolicyWindow();  
  // use default stream
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  CheckCudaErrors(cudaCtxResetPersistingL2Cache());
}

__global__ void HistogramGPUv1Kernel(const int *data, const uint32_t n, uint32_t *bin_counts) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const int bin_i = data[tid];
  atomicAdd(bin_counts + bin_i, 1);
}

void HistogramGPUv1(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *bin_counts) {
  const int threads = 1024;
  const int blocks = (n + threads - 1) / threads;
  CheckCudaErrors(cudaMemsetAsync(bin_counts, 0, sizeof(uint32_t)*kNBins, stream));
  HistogramGPUv1Kernel<<<blocks, threads, 0, stream>>>(data, n, bin_counts);
}

void PrepareHistogramGPUv2(const cudaStream_t &stream, uint32_t *bin_counts) {
  cudaDeviceProp prop;
  CheckCudaErrors(cudaGetDeviceProperties(&prop, 0));
  cout << "l2CacheSize:" <<  prop.l2CacheSize << endl;
  cout << "persistingL2CacheMaxSize:" <<   prop.persistingL2CacheMaxSize << endl;
  
  CheckCudaErrors(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize));
  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow = GetDefalutAccessPolicyWindow();

  stream_attribute.accessPolicyWindow.base_ptr  = (void *)bin_counts;
  const int num_bytes = min(prop.persistingL2CacheMaxSize, (int)sizeof(uint32_t)*kNBins);
  const float hit_ratio = 1.f; //max((float)num_bytes/(sizeof(uint32_t)*kNBins), 1.f);
  cout << "num_bytes:" << num_bytes << endl;
  cout << "hit_ratio:" << hit_ratio << endl;
  stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                  
  stream_attribute.accessPolicyWindow.hitRatio  = hit_ratio;
  stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
  stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
  
  // use default stream
  cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  CheckCudaErrors(cudaCtxResetPersistingL2Cache());
}

void HistogramGPUv2(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *bin_counts) {
  const int threads = 1024;
  const int blocks = (n + threads - 1) / threads;
  CheckCudaErrors(cudaMemsetAsync(bin_counts, 0, sizeof(uint32_t)*kNBins, stream));
  HistogramGPUv1Kernel<<<blocks, threads, 0, stream>>>(data, n, bin_counts);
}

__global__ void HistogramGPUv3Kernel(const int *data, const uint32_t n, uint32_t *bin_counts) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ uint32_t s_bin_counts[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  if (threadIdx.x < kNBins) {
    s_bin_counts[threadIdx.x] = 0;
  }
  cg::sync(cta);
  for (int i = tid; i < n; i += stride) {
    const int bin_i = data[i];
    atomicAdd(s_bin_counts + bin_i, 1);
  }
  cg::sync(cta);
  if (threadIdx.x < kNBins) {
    uint32_t sum = s_bin_counts[threadIdx.x];
    atomicAdd(bin_counts + threadIdx.x, sum);
  }
}

void HistogramGPUv3(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *bin_counts) {
  const int threads = 1024;
  const int blocks = (n + threads - 1) / threads;
  const size_t s_mem_size = sizeof(uint32_t)*kNBins;
  CheckCudaErrors(cudaMemsetAsync(bin_counts, 0, sizeof(uint32_t)*kNBins, stream));
  HistogramGPUv3Kernel<<<blocks, threads, s_mem_size, stream>>>(data, n, bin_counts);
}


__global__ void HistogramGPUv4Kernel(const int *data, const uint32_t n, uint32_t *tmp_bin_counts) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ uint32_t s_bin_counts[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tmp_bin_counts_offset = blockIdx.x * kNBins;
  const int stride = gridDim.x * blockDim.x;
  if (threadIdx.x < kNBins) {
    s_bin_counts[threadIdx.x] = 0;
  }
  cg::sync(cta);
  for (int i = tid; i < n; i += stride) {
    const int bin_i = data[i];
    atomicAdd(s_bin_counts + bin_i, 1);
  }
  cg::sync(cta);
  if (threadIdx.x < kNBins) {
    uint32_t sum = s_bin_counts[threadIdx.x];
    tmp_bin_counts[tmp_bin_counts_offset + threadIdx.x] = sum;
  }
}

__global__ void HistogramGPUv4MergeKernel(const uint32_t *tmp_bin_counts, const int n, uint32_t *bin_counts) {
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ uint32_t s_data[];

  uint32_t sum = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    sum += tmp_bin_counts[blockIdx.x + i * kNBins];
  }
  s_data[threadIdx.x] = sum; 
  for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);
    if (threadIdx.x < stride) {
      s_data[threadIdx.x] += s_data[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0) {
    bin_counts[blockIdx.x] = s_data[0];
  }
}

void HistogramGPUv4(const int *data, const uint32_t n, const cudaStream_t &stream, uint32_t *tmp_bin_counts, uint32_t *bin_counts) {
  const int threads = 1024;
  const int blocks = min((int)(n + threads - 1) / threads, kMaxBlocks);
  const size_t s_mem_size = sizeof(uint32_t)*kNBins;
  CheckCudaErrors(cudaMemsetAsync(bin_counts, 0, sizeof(uint32_t)*kNBins, stream));
  HistogramGPUv4Kernel<<<blocks, threads, s_mem_size, stream>>>(data, n, tmp_bin_counts);

  const int merge_threads = 256;
  HistogramGPUv4MergeKernel<<<kNBins, merge_threads, sizeof(uint32_t)*merge_threads, stream>>>(tmp_bin_counts, blocks, bin_counts);
}

