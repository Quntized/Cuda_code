#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <vector>

__tile_global__ void scale_3d(float* __restrict__ in, float* __restrict__ out, int width, int height,int depth){
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    in = ct::assume_aligned(in,16_ic);
    out = ct::assume_aligned(out , 16_ic);
    auto inSpan = ct::tensor_span{in, ct::extents{depth,height,width}};
    auto outSpan = ct::tensor_span{out, ct::extents{depth,height,width}};
    auto inView = ct::partition_view{inSpan, ct::shape{8_ic,8_ic,8_ic}};
    auto outView = ct::partition_view{outSpan, ct::shape{8_ic,8_ic,8_ic}};
    int bz = ct::bid().z;
    int by = ct::bid().y;
    int bx = ct::bid().x;
    printf("blocks_z %d, blocks_y %d, blocks_x %d",bz,by,bx);
    auto tile = inView.load_masked(bz,by,bx);
    auto scaled_tile = tile * 2.0f; //math . Also again same for the section 2.4 in cuda programming doc
    outView.store_masked(scaled_tile,bz,by,bx);
}



int main(int argc,char** argv){
    int depth = 4*8;
    int height = 8*8;
    int width = 12*8;
    int total_elements = depth*height*width;
    size_t bytes = sizeof(float) * total_elements;
    std::vector<float> h_in(total_elements,10.0f);
    std::vector<float> h_out(total_elements,0.0f);
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in , bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in,h_in.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,h_out.data(),bytes,cudaMemcpyHostToDevice);
    int blocks_x = (width + 8 - 1)/8;
    int blocks_y = (height + 8 - 1)/8;
    int blocks_z = (depth + 8 - 1)/8;
    std::cout<<"x = "<<blocks_x<<"y= "<<blocks_y<<"z = "<<blocks_z<<std::endl;
    std::cout<<"Launching Kernels = "<<"width - blocks_x ("<<blocks_x<<")"<<",height blocks_y ("<<blocks_y<<") , blocks_z depth ("<<blocks_z<<") \n";
    dim3 grid(blocks_x,blocks_y,blocks_z);
    scale_3d<<<grid,1>>>(d_in,d_out,width,height,depth);
    cudaMemcpy(h_out.data(),d_out,bytes,cudaMemcpyDeviceToHost);
    bool success = true;
    for(int i=0; i<total_elements; ++i){
        if(h_out[i] != 20.0f){
            std::cerr<<"ERROR!!"<<std::endl;
            success = false;
            break;
        }
    }
    if(success){
        std::cout<<"Congratulations"<<std::endl;
    }
    cudaFree(d_in);
    cudaFree(d_out);
}