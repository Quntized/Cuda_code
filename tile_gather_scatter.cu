#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_tile.h>

__tile_global__ void gather_scatter(float* data, int* indices, float* out){
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int,ct::shape<8>>;
    using f32x8 = ct::tile<float,ct::shape<8>>;
    data = ct::assume_aligned(data,16_ic);
    out = ct::assume_aligned(out,16_ic);
    indices = ct::assume_aligned(indices,16_ic);
    int bx = ct::bid().x;
    printf("block_index = %d \n",bx);
    auto linear_offset = bx*8+ct::iota<i32x8>();
    auto index_pts = indices + linear_offset;
    auto m_indices = ct::load(index_pts);
    auto data_ptrs = data+m_indices;
    auto generated_data = ct::load(data_ptrs);
    auto result = generated_data * 10.0f;
    auto out_ptrs = out+linear_offset;
    ct::store(out_ptrs,result);
}
int main(int argc, char** argv){
    int data_base_size = 100;
    size_t byte = sizeof(float);
    std::vector<float> h_data(data_base_size,0.0f);
    for(int i = 0; i<data_base_size; ++i){
        h_data[i] = i*1.0f;
    }
    std::vector<int> h_indices = {2,4,55,3,8,99,0,1};
    std::vector<float> h_out(8,0.0f);
    float *d_data,*d_out;
    int *d_indices;
    cudaMalloc(&d_data,data_base_size*byte);
    cudaMalloc(&d_out,8*byte);
    cudaMalloc(&d_indices,8*sizeof(int));
    cudaMemcpy(d_data,h_data.data(),byte*data_base_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,h_out.data(),byte*8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices,h_indices.data(),sizeof(int)*8,cudaMemcpyHostToDevice);

    gather_scatter<<<1,1>>>(d_data,d_indices,d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(),d_out,byte*8,cudaMemcpyDeviceToHost);
    for(int i =0; i<8; ++i){
        std::cout<<"result = "<<h_out[i]<<std::endl;
    }
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_out);
    return 0;
}