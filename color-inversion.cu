#include <cuda_runtime.h>
#include <stdio.h> 

__global__ void invert_kernel(unsigned char* image, int width, int height) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 

    if (tid < width*height) { 
        int idx = tid*4; 
        // invert all 4 pixles 
        image[idx] = 255-image[idx];
        image[idx+1] = 255-image[idx+1];
        image[idx+2] = 255-image[idx+2];
        // 4th one remains untouched  
    }
}


// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

