#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void kernelA(float* in,float* A,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        A[idx] = in[idx] + 1.0f;
    }
}
__global__ void kernelB(float* A,float* B,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        B[idx] = A[idx] * 2.0f;
    }
}
__global__ void kernelC(float* A,float* C,int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        C[idx] = A[idx] + 3.0f;
    }
}
__global__ void kernelD(float* B,float* C, float* out, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        out[idx] = B[idx] + C[idx];
    }
}
int main(int argc, char** argv){
    int N = 1 << 20;
    size_t size = sizeof(float)*N;
    printf("size in bytes = %zu \n",size);
    float *h_in,*h_out;
    cudaMallocHost(&h_in,size);
    cudaMallocHost(&h_out,size);
    for(int i=0; i<N; i++){
        h_in[i] = 1.0f;
    }
    float *d_in,*d_A,*d_B,*d_C,*d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);
    cudaMalloc(&d_out,size);
    cudaGraph_t graph;
    cudaGraphCreate(&graph,0);
    cudaGraphNode_t memcpyH2DNode;
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.srcPtr = make_cudaPitchedPtr(h_in,size,N,1);
    memcpyParams.dstPtr = make_cudaPitchedPtr(d_in,size,N,1);
    memcpyParams.extent = make_cudaExtent(size,1,1);
    memcpyParams.kind = cudaMemcpyHostToDevice;
    cudaGraphAddMemcpyNode(&memcpyH2DNode,graph,NULL,0,&memcpyParams);
    cudaGraphNode_t nodeA;
    cudaKernelNodeParams paramsA = {0};
    void* argsA[] = {&d_in,&d_A,&N};
    paramsA.func = (void*)kernelA;
    paramsA.gridDim = dim3((N + 256 - 1)/256);
    paramsA.blockDim = dim3(256);
    paramsA.kernelParams = argsA;
    cudaGraphNode_t depsA[] = {memcpyH2DNode};
    cudaGraphAddKernelNode(&nodeA,graph,depsA,1,&paramsA);
    cudaGraphNode_t nodeB,nodeC;
    cudaKernelNodeParams paramsB = paramsA;
    void* argsB[] = {&d_A,&d_B,&N};
    paramsB.func = (void*)kernelB;
    paramsB.kernelParams = argsB;
    cudaKernelNodeParams paramsC = paramsA;
    void* argsC[] = {&d_A,&d_C,&N};
    paramsC.func = (void*)kernelC;
    paramsC.kernelParams = argsC;
    cudaGraphNode_t depsBC[] = {nodeA};
    cudaGraphAddKernelNode(&nodeB,graph,depsBC,1,&paramsB);
    cudaGraphAddKernelNode(&nodeC,graph,depsBC,1,&paramsC);
    cudaGraphNode_t nodeD;
    cudaKernelNodeParams paramsD = paramsA;
    paramsD.func = (void*)kernelD;
    void* argsD[] = {&d_B,&d_C,&d_out,&N};
    paramsD.kernelParams = argsD;
    cudaGraphNode_t depsD[] = {nodeB,nodeC};
    cudaGraphAddKernelNode(&nodeD,graph,depsD,2,&paramsD);
    cudaGraphNode_t memcpyD2HNode;
    memcpyParams.srcPtr = make_cudaPitchedPtr(d_out,size,N,1);
    memcpyParams.dstPtr = make_cudaPitchedPtr(h_out,size,N,1);
    cudaGraphNode_t depsD2H[] = {nodeD};
    cudaGraphAddMemcpyNode(&memcpyD2HNode,graph,depsD2H,1,&memcpyParams);
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec,graph,NULL,NULL,0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraphLaunch(graphExec,stream);
    cudaStreamSynchronize(stream);
    std::cout<<"Execution complete, the result is : "<<h_out[0]<<std::endl;
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
//refer to section 4.2.2.1.1. Graph APIs
}