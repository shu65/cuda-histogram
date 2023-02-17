#pragma once
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CheckCudaErrors(err) __CheckCudaErrors(err, __FILE__, __LINE__)

inline void __CheckCudaErrors(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}