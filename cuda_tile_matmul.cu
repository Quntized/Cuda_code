#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <vector>
#include <cuda_fp16.h>

__tile_global__ void gemm(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C, std::size_t M , std::size_t K, std::size_t N){
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32_acc = ct::tile<float,ct::shape<32,32>>;
    A = ct::assume_aligned(A,16_ic);
    B = ct::assume_aligned(B,16_ic);
    C = ct::assume_aligned(C,16_ic);
    constexpr auto tm = 32_ic;
    constexpr auto tn = 32_ic;
    constexpr auto tk = 16_ic;
    auto aView = ct::partition_view{ct::tensor_span{A,ct::extents{M,K}},ct::shape{tm,tk}};
    auto bView = ct::partition_view{ct::tensor_span{B,ct::extents{K,N}}, ct::shape{tk,tn}};
    auto cView = ct::partition_view{ct::tensor_span{C,ct::extents{M,N}},ct::shape{tm,tn}};
    auto [bx,by,bz] = ct::bid();
    auto acc = ct::full<f32_acc>(0.0f);
    std::size_t num_k = (K +tk - 1)/tk;
    for(auto k : ct::irange(std::size_t{0},num_k)){
        acc = ct::mma(aView.load_masked(bx,k),bView.load_masked(k,by),acc);
    }
    cView.store_masked(acc,bx,by);
}
int main(int argc, char** argv){
    std::size_t M = 32;
    std::size_t K = 64;
    std::size_t N = 128;
    std::size_t size_A = M*K;
    std::size_t size_B = K*N;
    std::size_t size_C = M * N;
    std::vector<__half> h_A(size_A);
    std::vector<__half> h_B(size_B);
    std::vector<float> h_C(size_C,0.0f);
    /*if(argc >= 2){
        n = std::atoi(argv[1]);
    }*/
    for(int i=0; i<size_A; ++i){
        h_A[i] = __float2half(1.0f);
    }
    for(int i=0; i<size_B; ++i){
        h_B[i] = __float2half(1.0f);
    }
    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A,sizeof(__half)*size_A);
    cudaMalloc(&d_B,sizeof(__half)*size_B);
    cudaMalloc(&d_C,sizeof(float)*size_C);
    cudaMemcpy(d_A,h_A.data(),sizeof(__half)*size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B.data(),sizeof(__half)*size_B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C.data(),sizeof(float)*size_C,cudaMemcpyHostToDevice);
    int blocks_x = (M+32-1)/32;
    int blocks_y = (N+32-1)/32;
    dim3 grid(blocks_x,blocks_y,1);
    std::cout<<"Launching Tensor Kernel with GEMM kernel..\n"<<std::endl;
    std::cout<<"Grid ( "<<blocks_x<<", "<<blocks_y<<"). "<<std::endl;
    gemm<<<grid,1>>>(d_A,d_B,d_C,M,K,N);
    std::cout << "2. Kernel launched. Waiting for GPU to finish...\n" << std::endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Launch error: "
              << cudaGetErrorString(err) << std::endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Runtime error: "
              << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(),d_C,sizeof(float)*size_C,cudaMemcpyDeviceToHost);
    bool success = true;
    for(std::size_t i=0; i<size_C; ++i){
        if(h_C[i] != 64.0f){
            std::cerr<<"Somehow got error. "<<std::endl;
            success = false;
            break;
        }
    }
    if(success){
        std::cout<<"Success. "<<std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
//I think this should be test again;
}