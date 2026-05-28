#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 

    if (tid < N) {
        C[tid] = A[tid]+B[tid];
    } 
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

/***
 * 
 * You are launching a kernel with 256 threads per block and having ceil(N/256) blocks
 * Each block will be assigned to some SM (streaming multiprocessor) and executed. We are using one thread per index to sum the elements
 * each block will be executed in single step and in fact each block is split into 32 threads and executed as a SIMT 
 * 
 * tid is the thread id 
 * 
 * cudaDeviceSynchronize() ensures that all the launched block of threads are finished before exiting 
 * 
 * 
 */