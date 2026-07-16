#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <vector>

__tile_global__ void conditional_loop(float* __restrict__ arr,float* __restrict__ out,int N){
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float,ct::shape<8>>;
    arr = ct::assume_aligned(arr,16_ic);
    out = ct::assume_aligned(out,16_ic);
    auto inView = ct::partition_view{ct::tensor_span{arr,ct::extents{N}},ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}},ct::shape{8_ic}};
    int bx = ct::bid().x;
    int nbx = ct::num_blocks().x;
    auto tile = ct::full<f32x8>(0.0f);
    if(bx < nbx - 1){
        printf("bx = %d \n",bx);
        tile = inView.load(bx);
    }
    outView.store_masked(tile,bx);
}
int main(int argc,char** argv){
    int n = 24;
    if(argc >= 2){
        n = std::atoi(argv[1]);
    }
    std::vector<float> h_arr(n,12.0f);
    std::vector<float> h_out(16,0.0f);
    float *d_arr, *d_out;
    cudaMalloc(&d_arr,n*sizeof(float));
    cudaMalloc(&d_out,16*sizeof(float));
    cudaMemcpy(d_arr, h_arr.data(),sizeof(float)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,h_out.data(),sizeof(float)*16, cudaMemcpyHostToDevice);
    conditional_loop<<<3,1>>>(d_arr,d_out,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(),d_out,sizeof(float)*16,cudaMemcpyDeviceToHost);
    std::cout<<"Size = "<<h_out.size()<<std::endl;
}