#include <stdexcept>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

#include "check.cuh"
#include "util.cuh"

__global__ void cuda_check_kernel_invocation(bool print) {
  if (print) {
    printf("GPU kernel invocation works!\n");
  }
}

bool works(bool print) {
  cudaError_t err;

  cuda_check_kernel_invocation<<<1, 1>>>(print);

  err = cudaPeekAtLastError();
  gpuAssert(err, __FILE__, __LINE__, false);
  if (err != cudaSuccess) {
    return false;
  }

  err = cudaDeviceSynchronize();
  gpuAssert(err, __FILE__, __LINE__, false);
  if (err != cudaSuccess) {
    return false;
  }

  return true;
}

bool have_gpu() {
  int deviceCount = 0;
  CUresult error;

  error = cuInit(0);
  if (error != CUDA_SUCCESS) {
    return false;
  }

  error = cuDeviceGetCount(&deviceCount);
  if (error != CUDA_SUCCESS) {
    throw std::runtime_error("Failed to query the number of CUDA devices (" +
                             std::string(std::strerror(static_cast<int>(error))) + ")\n");
  }

  return deviceCount > 0;
}
