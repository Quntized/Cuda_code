//refer to https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html
#include <iostream>
#include <cuda_runtime.h>
__global__ void kernel1(float* data, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size){
        data[idx] = 1.0f;
    }
}
__global__ void kernel2(float* data, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size){
        data[idx] *=2.0f;
    }
}
int main() {
    int N = 10000;
    size_t size = N*sizeof(float);
    cudaStream_t stream_allocator,stream_reader;
    cudaEvent_t mmry_ready_event;
    cudaStreamCreate(&stream_allocator);
    cudaStreamCreate(&stream_reader);
    cudaEventCreate(&mmry_ready_event);
    float* d_density = nullptr;
    cudaMallocAsync(&d_density, size, stream_allocator);
    int threads = 512;
    int blocks = (N + threads - 1)/threads;
    kernel1<<<blocks,threads,0,stream_allocator>>>(d_density,N);
    cudaEventRecord(mmry_ready_event,stream_allocator);
    cudaStreamWaitEvent(stream_reader,mmry_ready_event,0);
    kernel2<<<blocks,threads,0,stream_reader>>>(d_density,N);
    cudaFreeAsync(d_density,stream_reader);
    cudaStreamSynchronize(stream_reader);
    cudaStreamDestroy(stream_allocator);
    cudaStreamDestroy(stream_reader);
    cudaEventDestroy(mmry_ready_event);
    //cudaFreeAsync(d_density,stream_allocator);
    return 0;
}
