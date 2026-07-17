#include <stdio.h>
#include <cuda_runtime.h>

__global__ void func()
{
#if __CUDA_ARCH__ >= 800
printf("Success! The CUDA Architecture ID is: %d\n", __CUDA_ARCH__);
#elif __CUDA_ARCH__ >= 700
   // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
   // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
   // Device code path for compute capability 5.x
#elif !defined(__CUDA_ARCH__)
   // Host code path
#endif //taken from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions
}
int main(int argc, char** argv){
    func<<<1,1>>>();
    cudaDeviceSynchronize(); //funnier here must synchronize otherwise the program will close before gpu has time to flush it's printf :')
}