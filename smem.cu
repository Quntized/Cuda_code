#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#define THREAD_PER_BLOCK_X 32
#define THREAD_PER_BLOCK_Y 32
#define INDX(row,col,ld) (((col) * (ld)) + (row))

__global__ void smem_transpose(int m, float* a, float* b)
{
    __shared__ float smemArray[THREAD_PER_BLOCK_X][THREAD_PER_BLOCK_Y];
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;
    if (row < m && col < m)
    {
        smemArray[threadIdx.x][threadIdx.y] = a[INDX(tileX + threadIdx.x , tileY + threadIdx.y, m)];
    }
    __syncthreads();
    if(row < m && col < m)
    {
        b[INDX(tileY + threadIdx.x, tileX + threadIdx.y , m)] = smemArray[threadIdx.y][threadIdx.x];
    }
    return;
}
void verifyTranspose(const std::vector<float>& original, const std::vector<float>& transposed, int m)
{
    for (int row = 0; row<m; row++){
        for (int col=0; col<m; col++){
            if(original[INDX(row,col,m)] != transposed[INDX(col,row,m)]){
                std::cerr<<"Mismatch found at row "<<row<<", col "<<col<<"\n";
                return;
            }
        }
    }
    std::cout<<"success!!!\n";
}
int main(int argc, char** argv){
    int m =2048;
    if (argc>=2){
        m = std::atoi(argv[1]);
    }
    size_t bytes = m*m*sizeof(float);
    std::vector<float> h_a(m*m);
    std::vector<float> h_c(m*m,0.0f);
    for (int i=0; i<m*m; ++i){
        h_a[i] = static_cast<float>(i);
    }
    float *d_a, *d_c;
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_c,bytes);
    cudaMemcpy(d_a,h_a.data(),bytes, cudaMemcpyHostToDevice);
    dim3 threads(THREAD_PER_BLOCK_X,THREAD_PER_BLOCK_Y);
    dim3 blocks((m + threads.x - 1)/threads.x, (m +threads.y - 1)/threads.y);
    std::cout<<"Launching Kernel with a "<< m <<" x  "<<m<<" matrix \n";
    std::cout<<blocks.x<<" "<<blocks.y<<"\n";
    std::cout<<threads.x<<" "<<threads.y<<"\n";
    smem_transpose<<<blocks,threads>>>(m,d_a,d_c);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel Launch Error: " << cudaGetErrorString(launchErr) << "\n";
        return -1;
    }
    cudaDeviceSynchronize();
    cudaError_t syncErr = cudaGetLastError();
    if (syncErr != cudaSuccess) {
        std::cerr << "Kernel Execution Error: " << cudaGetErrorString(syncErr) << "\n";
        return -1;
    }
    cudaMemcpy(h_c.data(),d_c,bytes, cudaMemcpyDeviceToHost);
    verifyTranspose(h_a,h_c,m);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}