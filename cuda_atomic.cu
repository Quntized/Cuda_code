#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cuda/atomic>

__global__ void sumReduction(int n, float* array, float* result)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n)
    {
        printf("threadIdx.x %d , blockIdx.x %d, blockDim.x %d.",threadIdx.x,blockIdx.x,blockDim.x);
        cuda::atomic_ref<float,cuda::thread_scope_device> result_ref(*result);
        result_ref.fetch_add(array[tid]);
    }
}
int main(int argc, char** argv)
{
    int n= 1024;
    size_t bytes = n * sizeof(float);
    std::vector<float> h_array(n,1.0f);
    float h_result = 0.0f;
    float *d_array, *d_result;
    cudaMalloc(&d_array,bytes);
    cudaMalloc(&d_result,sizeof(float));
    cudaMemcpy(d_array,h_array.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    int threadsperblock =256;
    int blocks= (n + threadsperblock - 1)/threadsperblock;
    std::cout<<"Threadblock = "<<threadsperblock<<std::endl;
    std::cout<<"blocks = "<<blocks<<std::endl;
    std::cout<<"Start "<<n<<"\n";
    sumReduction<<<blocks,threadsperblock>>>(n,d_array,d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result,sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"this ==== "<<h_result<<std::endl;
    if(h_result == (float)n){
        std::cout<<"Success!!!! "<<std::endl;
    }
    cudaFree(d_array);
    cudaFree(d_result);
}