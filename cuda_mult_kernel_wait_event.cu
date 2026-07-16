#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
__global__ void gpu_call1(float* A, int N){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id < N){
        A[id] = A[id] * 2.0f;
    }
}
__global__ void gpu_call2(float* B,int N){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id < N){
        B[id] = B[id] + 2.0f;
    }
}
void dependentCPUtask(){
    std::cout<<"cpu doing heavy task "<<std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    std::cout<<"cpu work done."<<std::endl;
}
int main(){
    int N = 1 >> 20;
    size_t size = N*sizeof(float);
    std::vector<float> h_A(N,1.0f);
    std::vector<float> h_B(N,2.0f);
    float *d_A,*d_B;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    int threadspb = 256;
    int blockspg = (N + threadspb - 1)/threadspb;
    dim3 t(threadspb);
    dim3 g(blockspg);
    cudaMemcpy(d_A,h_A.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B.data(),size,cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaEvent_t event;
    cudaStreamCreate(&stream);
    cudaEventCreate(&event);
    gpu_call1<<<t,g,0,stream>>>(d_A,N);
    cudaEventRecord(event,stream);
    gpu_call2<<<t,g,0,stream>>>(d_B,N);
    cudaEventSynchronize(event);
    dependentCPUtask();
    cudaStreamSynchronize(stream);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}