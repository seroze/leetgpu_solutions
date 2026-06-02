#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void dot_product_kernel(
    const half* A,
    const half* B,
    float* sum,
    int N)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N)
    {
        float a = __half2float(A[idx]);
        float b = __half2float(B[idx]);

        sdata[tid] = a * b;
    }
    else
    {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void convert_to_half(
    const float* sum,
    half* result)
{
    result[0] = __float2half(sum[0]);
}

extern "C" void solve(
    const half* A,
    const half* B,
    half* result,
    int N)
{
    const int THREADS_PER_BLOCK = 256;

    int blocksPerGrid =
        (N + THREADS_PER_BLOCK - 1)
        / THREADS_PER_BLOCK;

    float* d_sum;

    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    dot_product_kernel<<<blocksPerGrid,
                         THREADS_PER_BLOCK>>>(
        A,
        B,
        d_sum,
        N);

    convert_to_half<<<1, 1>>>(
        d_sum,
        result);

    cudaDeviceSynchronize();

    cudaFree(d_sum);
}