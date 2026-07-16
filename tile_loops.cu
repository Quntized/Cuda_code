#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_tile.h>

__tile_global__ void conditional_load(float* __restrict__ arr, float* __restrict__ out, int num_tiles, int N){
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float,ct::shape<8>>;
    arr = ct::assume_aligned(arr,16_ic);
    out = ct::assume_aligned(out,16_ic);
    auto inView = ct::partition_view{ct::tensor_span{arr,ct::extents{8*num_tiles}},ct::shape{8_ic}};
    auto outView =ct::partition_view{ct::tensor_span{out,ct::extents{8*num_tiles}},ct::shape{8_ic}};
    int bx = ct::bid().x;
    printf("block_idx %d \n",bx);
    int nbx = ct::num_blocks().x;
    printf("total number of blocks in x dir %d \n",nbx);
    auto acc = ct::full<f32x8>(0.0f);
    for(auto k : ct::irange(0,num_tiles)){
        auto tile = inView.load(k);
        acc = acc + tile;
    }
    outView.store(acc,bx);

}
int main(int argc, char** argv){
    int n= 20;
    int num_tiles = 2;
    std::vector<float> h_arr(n);
    for(int i=0; i<n; ++i){
        h_arr[i]=static_cast<float>(i);
    }
    std::vector<float> h_out(n,-99.0f);
    float *d_arr,*d_out;
    cudaMalloc(&d_arr,sizeof(float)*n);
    cudaMalloc(&d_out,sizeof(float)*n);
    cudaMemcpy(d_arr, h_arr.data(),sizeof(float)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out.data(),sizeof(float)*n,cudaMemcpyHostToDevice);
    int num_blocks  = (n + 8 - 1)/8;
    std::cout<<"num_blocks = "<<num_blocks<<std::endl;
    conditional_load<<<num_blocks,1>>>(d_arr,d_out,num_tiles,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(),d_out,sizeof(float)*n,cudaMemcpyDeviceToHost);
    for(int i =0; i<20; ++i){
        std::cout<<"result = "<<h_out[i]<<std::endl;
    }
    cudaFree(d_arr);
    cudaFree(d_out);

}