#include <cuda_runtime.h>

__device__ __forceinline__ float gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}

__global__ void geglu_kernel(
    const float* input,
    float* output,
    int halfN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < halfN) {

        float left = input[idx];
        float right = input[idx + halfN];

        output[idx] = left * gelu(right);
    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
