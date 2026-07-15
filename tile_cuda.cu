#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_tile.h>
#include <cstdio>

namespace ct = cuda::tiles;
constexpr int TILE_SIZE = 32;
using FloatTile = ct::tile<float, ct::shape<TILE_SIZE>>;
using IntTile = ct::tile<int,ct::shape<TILE_SIZE>>;

__tile_global__ void vector_add_tile(float* a, float* b, float* c)
{
    int block_idx = ct::bid().x;
    printf("block index , %d",block_idx);
    int mem_offset = block_idx*TILE_SIZE;
    auto local_offset = ct::iota<IntTile>();
    auto ptrs_a = a + mem_offset + local_offset;
    auto ptrs_b = b + mem_offset + local_offset;
    auto ptrs_c = c + mem_offset + local_offset;
    auto tile_a  = ct::load(ptrs_a);
    auto tile_b = ct::load(ptrs_b);
    auto tile_c = tile_a + tile_b;
    ct::store(ptrs_c,tile_c);
}

int main(int argc, char** arg)
{
    int n = 1024;
    size_t bytes = n*sizeof(float);
    std::vector<float> h_a(n,2.0f);
    std::vector<float> h_b(n,3.0f);
    std::vector<float> h_c(n,0.0f);
    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,h_c.data(),bytes,cudaMemcpyHostToDevice);
    int num_blocks = n / TILE_SIZE;
    vector_add_tile<<<num_blocks,1>>>(d_a,d_b,d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost);
    bool Success = true;
    for(int i=0; i<n; ++i){
        if(h_c[i]!=5.0f){
            Success = false;
            break;
        }
    }
    if(!Success){
        std::cerr<<"Error occurred!"<<std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}