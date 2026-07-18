// taken from https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_18656f53eb2a7e54500f6fb95a830b47d
#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess){ \
            std::cerr<<"CUDA error at "<<__FILE__<<" : "<<__LINE__ \
            <<" code= "<<err<<" ( "<<cudaGetErrorString(err)<<" ) \n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
int main(){
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout<<"Device count: "<<deviceCount<<std::endl;
    if(deviceCount == 0){
        std::cout<<"No CUDA-capable devices \n";
    }
    for (int dev=0; dev<deviceCount; ++dev){
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
        std::cout<<"Device "<<dev<<". name: "<<prop.name<<std::endl;
        std::cout<<"cc major: "<<prop.major<<".cc minor "<<prop.minor<<"\n";
        double memGB = static_cast<double>(prop.totalGlobalMem)/(1024*1024*1024);
        std::cout<<"Total glob mem "<<std::fixed<<std::setprecision(2)<<memGB<<"\n";
        std::cout<<"SM count "<<prop.multiProcessorCount<<"\n";
        std::cout<<"Max threads per SM= "<<prop.maxThreadsPerMultiProcessor<<"\n";
        std::cout<<"Max threads per block "<<prop.maxThreadsPerBlock<<"\n";
        std::cout<<"Warp Size: "<<prop.warpSize<<"\n";
        std::cout<<"Max shared mem per SM "<<prop.sharedMemPerMultiprocessor/1024<<" in KB"<<"\n";
        std::cout<<"shared mem per block "<<prop.sharedMemPerBlock/1024<<" in KB\n";
        std::cout<<"regs per block "<<prop.regsPerBlock<<"\n";
        std::cout<<"regs per SM "<<prop.regsPerMultiprocessor<<"\n";
        //std::cout<<"Num regs "<<prop.numRegs<<"\n"; 
        int maxTh = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        std::cout<<"max current thread the hardware can run "<<maxTh<<std::endl;
        return 0;
    }
}