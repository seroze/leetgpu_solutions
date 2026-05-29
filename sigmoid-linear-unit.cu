#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__
void silu_kernel(const float* input, float* output, int N) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {

        float x = input[tid];

        output[tid] = x * sigmoid(x);
    }
}

// input, output are device pointers
extern "C"
void solve(const float* input, float* output, int N) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input,
        output,
        N
    );

    cudaDeviceSynchronize();
}