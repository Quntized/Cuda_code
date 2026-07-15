#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void CheckCuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error during " << action << ": "
              << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

__global__ void SquareKernel(const int* input, int* output, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * input[idx];
  }
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  int test_cases = 0;
  if (!(std::cin >> test_cases)) {
    return 0;
  }

  for (int case_index = 1; case_index <= test_cases; ++case_index) {
    int n = 0;
    std::cin >> n;

    std::vector<int> host_input(n);
    for (int i = 0; i < n; ++i) {
      std::cin >> host_input[i];
    }

    std::vector<int> host_output(n);
    if (n > 0) {
      int* device_input = nullptr;
      int* device_output = nullptr;

      const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(int);
      CheckCuda(cudaMalloc(&device_input, bytes), "cudaMalloc(device_input)");
      CheckCuda(cudaMalloc(&device_output, bytes), "cudaMalloc(device_output)");

      CheckCuda(
          cudaMemcpy(device_input, host_input.data(), bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy host to device");

      constexpr int kThreadsPerBlock = 256;
      const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
      SquareKernel<<<blocks, kThreadsPerBlock>>>(device_input, device_output, n);
      CheckCuda(cudaGetLastError(), "kernel launch");
      CheckCuda(cudaDeviceSynchronize(), "kernel execution");

      CheckCuda(
          cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy device to host");

      CheckCuda(cudaFree(device_input), "cudaFree(device_input)");
      CheckCuda(cudaFree(device_output), "cudaFree(device_output)");
    }

    std::cout << "Case #" << case_index << ':';
    for (int value : host_output) {
      std::cout << ' ' << value;
    }
    std::cout << '\n';
  }

  return 0;
}
