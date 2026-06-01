#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < N && row < N ) {
        int newIdx = N*row + col; 
        B[newIdx] = A[newIdx];
    }

}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    // int threadsPerBlock = 16*16;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);
    // int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    // copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
