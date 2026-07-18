#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void gpu_calling(const float* A, const float* B,float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        C[i] = A[i] + B[i] + (A[i] *0.00001f);
    }
}
int main(){
    int N = 1 << 20; //2^20;
    size_t size = N * sizeof(float);
    std::vector<float> h_A(N,1.0f);
    std::vector<float> h_B(N,2.0f);
    std::vector<float> h_C(N,0.0f);
    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C.data(),size,cudaMemcpyHostToDevice);
    int threads_perblock = 256;
    int blocks_per_grid = (N + threads_perblock - 1)/threads_perblock;
    dim3 grid(blocks_per_grid);
    dim3 block(threads_perblock);
    std::cout<<"Asynchronous case: "<<std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,stream);
    gpu_calling<<<grid,block>>>(d_A,d_B,d_C,N);
    cudaEventRecord(stop,stream);
    cudaStreamSynchronize(stream);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    std::cout<<"Kernel execution time: "<<elapsedTime<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}