#include <cuda_runtime.h>

#define TILE 16

__global__ void identity_kernel(float* mat, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N) {
        int row = idx / N;
        int col = idx % N;
        mat[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0f;

    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {

        int aCol = tile * TILE + threadIdx.x;
        int bRow = tile * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N)
                ? A[row * N + aCol]
                : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N)
                ? B[bRow * N + col]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += (double)As[threadIdx.y][k] *
                   (double) Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = (float)sum;
}

extern "C" void solve(
    const float* input,
    float* output,
    int N,
    int P)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    int threads = 256;
    int blocks = ((long long)N * N + threads - 1) / threads;

    // A^0 = I
    if (P == 0) {
        identity_kernel<<<blocks, threads>>>(output, N);
        cudaDeviceSynchronize();
        return;
    }

    // A^1 = A
    if (P == 1) {
        cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
        return;
    }

    float* base;
    float* result;
    float* temp;

    cudaMalloc(&base, bytes);
    cudaMalloc(&result, bytes);
    cudaMalloc(&temp, bytes);

    cudaMemcpy(base, input, bytes, cudaMemcpyDeviceToDevice);

    identity_kernel<<<blocks, threads>>>(result, N);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE);

    while (P > 0) {

        if (P & 1) {

            matmul_kernel<<<gridDim, blockDim>>>(
                result,
                base,
                temp,
                N);

            float* swap = result;
            result = temp;
            temp = swap;
        }

        matmul_kernel<<<gridDim, blockDim>>>(
            base,
            base,
            temp,
            N);

        float* swap = base;
        base = temp;
        temp = swap;


        P >>= 1;

    }

    cudaMemcpy(output, result, bytes, cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    cudaFree(base);
    cudaFree(result);
    cudaFree(temp);
}
