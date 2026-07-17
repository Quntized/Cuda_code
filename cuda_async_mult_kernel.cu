#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <thread>

__global__ void kernel1(float* A, int N){
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < N){
        A[indx] = A[indx] + 2.0f; //doing math here;
    }
}

__global__ void kernel2(float* B, int N){
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < N){
        B[indx] = B[indx] * 2.0f; //doing math here; same
    }
}
void func1(){
    std::cout<<"Started CPU Work func1"<<std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    std::cout<<"Finished 1"<<std::endl;
}
void func2(){
    std::cout<<"Started CPU Work func2"<<std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    std::cout<<"Finished 2"<<std::endl;
}
void nextfunc(){
    std::cout<<"Started CPU Work next"<<std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    std::cout<<"Finished next work"<<std::endl;
}
bool allCPUWorkDone(){
    static bool cpufuncfinished1 = false;
    static bool cpufuncfinished2 = false;
    if(not cpufuncfinished1){
        std::cout<<"Calling func1"<<std::endl;
        func1();
        std::cout<<"leaving func1"<<std::endl;
        cpufuncfinished1 =true;
    }
    if(not cpufuncfinished2){
        std::cout<<"Calling func2"<<std::endl;
        func2();
        std::cout<<"leaving func2"<<std::endl;
        cpufuncfinished2 =true;
    }
    return (cpufuncfinished1 && cpufuncfinished2);
}
void doNextChunkOfCPUWork(){
    std::cout<<"Calling next cpu work"<<std::endl;
    nextfunc();
    std::cout<<"Finished next cpu work"<<std::endl;

}

int main(int argc , char** argv){
    int N = 1 << 20;
    printf("The array size = %d \n",N);
    //std::vector<float> h_A(N,20.0f);
    //std::vector<float> h_B(N,12.0f); //the array shouldn't be the concern , the understanding behind the section 2.5.3.4 is concern.
    float *d_A,*d_B;
    //cudaMallocHost(&h_A,sizeof(float)*N);  //not cudaMallocHost but can be used as cudaHostRegister(h_A.data(), size, cudaHostRegisterDefault); and at the end cudaHostUnregister(h_A.data()); for unpin
    float *h_A,*h_B;
    cudaMallocHost(&h_A,sizeof(float)*N);

    cudaMallocHost(&h_B,sizeof(float)*N);

    for(int i = 0; i<N; ++i){
        h_A[i] = 20.0f;
    } //same for this;
    for(int i = 0; i<N; ++i){
        h_B[i] = 2.0f;
    }
    cudaMalloc(&d_A,sizeof(float)*N);
    cudaMalloc(&d_B,sizeof(float)*N);
    int t = 256;
    int b = (N + t - 1)/t;
    dim3 th(t);
    dim3 gr(b);
    cudaMemcpy(d_A,h_A,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaEvent_t event;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    bool copyStarted = false;
    cudaEventCreate(&event);
    kernel1<<<gr,th,0,stream1>>>(d_A,N);
    cudaEventRecord(event,stream1);
    kernel2<<<gr,th,0,stream1>>>(d_B,N);
    while (not allCPUWorkDone() || not copyStarted){
        doNextChunkOfCPUWork();
        if(not copyStarted){
            if(cudaEventQuery(event) == cudaSuccess){
                cudaMemcpyAsync(h_A,d_A,sizeof(float)*N,cudaMemcpyDeviceToHost,stream2);
                copyStarted = true;
            }
        }
    }
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
