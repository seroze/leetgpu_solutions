#include <cuda_runtime.h>

__global__ void count_kernel(const int* input, int* output, int N, int K){
    int tid = threadIdx.x; 
    int idx = blockDim.x * blockIdx.x + tid; 
    
    int stride = blockDim.x * gridDim.x; 
    __shared__ int block_count;

    if (threadIdx.x == 0)
        block_count = 0;

    __syncthreads(); // wait for all of them to finish

    int local = 0; // local variable 
    for (int i = idx; i<N; i+= stride) {
        if (input[i] == K) local += 1;
    }

    if (local)
        atomicAdd(&block_count, local);

    __syncthreads(); // wait for everyone to finish

    if (threadIdx.x == 0)
        atomicAdd(output, block_count);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {

    cudaMemset(output, 0, sizeof(int));

    constexpr int BLOCK_SIZE = 256;

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Avoid launching absurd numbers of blocks
    blocks = min(blocks, 65535);

    count_kernel<<<blocks, BLOCK_SIZE>>>(input, output, N, K);
}

