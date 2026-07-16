#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_tile.h>

__tile_global__ void gather_safe(int* __restrict__ arr, int* __restrict__ out , int N){
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    using i32x8 = ct::tile<int, ct::shape<8>>;
    int bx = ct::bid().x;
    printf("block indes = %d \n",bx);
    auto offsets = bx*8 + ct::iota<i32x8>();
    auto mask = offsets < N;
    auto ptrs = arr + offsets;
    auto tile = ct::load_masked(ptrs,mask,0);
    ct::store_masked(out+offsets,tile,mask);
}
int main(int argc, char** argv){
    int n = 20;
    std::vector<int> h_arr(n);
    for(int i=0; i<n; ++i){
        h_arr[i] = i;
    }
    std::vector<int> h_out(n);
    for(int i=0; i<n; ++i){
        h_out[i] = -99;
    }
    int *d_arr,*d_out;
    cudaMalloc(&d_arr,sizeof(int)*n);
    cudaMalloc(&d_out,sizeof(int)*n);
    cudaMemcpy(d_arr,h_arr.data(),sizeof(int)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,h_out.data(),sizeof(int)*n,cudaMemcpyHostToDevice);
    int num_blocks = (n + 8 - 1)/8;
    gather_safe<<<num_blocks,1>>>(d_arr,d_out,16);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(),d_out,sizeof(int)*n,cudaMemcpyDeviceToHost);
    for(int i = 10; i<20; ++i){
        std::cout<<"result = "<<h_out[i]<<std::endl;
    }
    cudaFree(d_arr);
    cudaFree(d_out);

}