#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess){ \
            std::cerr<<"CUDA error at "<<__FILE__<<" : "<<__LINE__ \
            <<" code= "<<err<<" ( "<<cudaGetErrorString(err)<<" ) \n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
__global__ void kernel(float* data){
    __shared__ float shared_cache[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];
    shared_cache[threadIdx.x] = val * 2.0f;
    __syncthreads();
    data[idx] = shared_cache[threadIdx.x] + 1.0f;
}
int main(){
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,reinterpret_cast<const void*>(kernel)));
    std::cout<<"ptx virtual arch: "<<attr.ptxVersion<<"\n";
    std::cout<<"Binary arch: "<<attr.binaryVersion<<"\n";
    std::cout<<"Registers per thread: "<<attr.numRegs<<"\n";
    std::cout<<"Static Shared Memory(bytes): "<<attr.sharedSizeBytes<<"\n";
    std::cout<<"Max threads per block: "<<attr.maxThreadsPerBlock<<"\n";
    std::cout<<"Prefered shm carveout: "<<attr.preferredShmemCarveout<<"\n";
    const size_t regs_per_thread = attr.numRegs;
    const size_t allctd_reg_per_thrd =  ((regs_per_thread + 8 - 1)/8);
    std::cout<<"allocated_registration_per_threads: "<<allctd_reg_per_thrd * 8<<std::endl;
    int desired_carveout = 100; //100% used for shm;
    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),cudaFuncAttributePreferredSharedMemoryCarveout,desired_carveout));
    cudaFuncAttributes upd_attr;
    CUDA_CHECK(cudaFuncGetAttributes(&upd_attr,reinterpret_cast<const void*>(kernel)));
    std::cout<<"updated shmem carveout: "<<upd_attr.preferredShmemCarveout<<"\n";
    return 0;
}