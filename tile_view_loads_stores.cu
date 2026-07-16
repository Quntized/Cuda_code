#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_tile.h>

__tile_global__ void vec_add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out)
{
    namespace ct = cuda::tiles;
    //cuda::tiles::__1::tensor_span<float, cuda::tiles::__1::extents<unsigned int, 128UL>, cuda::tiles::__1::layout_right, cuda::tiles::__1::default_accessor<float>>
    using namespace ct::literals;
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    out = ct::assume_aligned(out, 16_ic);
    auto aSpan = ct::tensor_span{a, ct::extents{128_ic}}; //refer to Writing Tile Kernels section 2.4 of cuda programming model for better understanding;
    auto bSpan = ct::tensor_span{b, ct::extents{128_ic}};
    //printf("%d",aSpan);
    auto oSpan = ct::tensor_span{out, ct::extents{128_ic}};
    auto aView = ct::partition_view{aSpan, ct::shape{8_ic}};
    auto bView = ct::partition_view{bSpan, ct::shape{8_ic}};
    auto oView = ct::partition_view{oSpan , ct::shape{8_ic}};
    int bx = ct::bid().x;
    printf("block idx = %d \n",bx);
    auto aTile = aView.load(bx);
    auto bTile = bView.load(bx);
    oView.store(aTile+bTile, bx); //this can also work;
/*    auto oTile = aTile + bTile;
    for(int i =0; i<8; ++i){
        float a_val = ct::get<0>(aTile, i);
        float b_val = ct::get<0>(bTile,i);
        float o_val = ct::get<0>(oTile,i);
        printf("Element [%d]: %f + %f = %f\n", i, a_val, b_val, o_val);
    }  */
}
int main(int argc, char** argv){
    int n = 128;
    size_t bytes = n * sizeof(float);
    std::vector<float> h_a(n,10.0f);
    std::vector<float> h_b(n,11.0f);
    std::vector<float> h_out(n,0.0f);
    float* d_a, *d_b, *d_out;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out,bytes);
    cudaMemcpy(d_a,h_a.data(),bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,h_out.data(),bytes, cudaMemcpyHostToDevice);
    int num_blocks = n/8;
    std::cout<<"Launching kernel with : "<<num_blocks<<std::endl;
    vec_add<<<num_blocks,1>>>(d_a,d_b,d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(),d_out,bytes,cudaMemcpyDeviceToHost);
    bool success = true;
    for(int i =0; i<n; ++i){
        if(h_out[i] != 21.0f){
            std::cerr<<"Mismatch the result what you have desired...."<<std::endl;
            success = false;
            break;
        }
    }
    if(success){
        std::cout<<"Congratulations!! "<<std::endl;
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return 0;
}
