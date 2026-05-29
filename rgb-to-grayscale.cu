#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int total_pixels = width * height;

    if (tid < total_pixels) {

        int idx = tid * 3;

        float r = input[idx];
        float g = input[idx + 1];
        float b = input[idx + 2];

        output[tid] =
            0.299f * r +
            0.587f * g +
            0.114f * b;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}

