#include <cuda_runtime.h>

__global__ void dotproduct(const float* A, const float* B, float* result, int N){

    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx<N){
        atomicAdd(result, A[idx]*B[idx]);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {

    int threadsPerBlock = 512;
    int noOfBlocks = (N+threadsPerBlock-1)/threadsPerBlock; 

    cudaMemset(result, 0, sizeof(float));
    dotproduct<<<noOfBlocks, threadsPerBlock>>>(A, B, result, N);
}
