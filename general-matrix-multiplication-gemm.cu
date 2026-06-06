#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void gemm_kernel(
    const half* A,
    const half* B,
    half* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta)
{
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over K dimension tiles
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Load tile from A
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        // Load tile from B
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum +=
                __half2float(As[threadIdx.y][k]) *
                __half2float(Bs[k][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {

        int idx = row * N + col;

        float c_old = __half2float(C[idx]);

        float result =
            alpha * sum +
            beta * c_old;

        C[idx] = __float2half(result);
    }
}

extern "C" void solve(
    const half* A,
    const half* B,
    half* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta)
{
    dim3 threadsPerBlock(TILE, TILE);

    dim3 blocksPerGrid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE);

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A,
        B,
        C,
        M,
        N,
        K,
        alpha,
        beta);
}
