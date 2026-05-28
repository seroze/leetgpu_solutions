#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    if (tid<N/2) {
        int idx = tid;
        int swap_idx = N-1-idx; 
        // swap tid and N-tid 
        float temp = input[swap_idx];
        input[swap_idx] = input[idx];
        input[idx] = temp;
    }

}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N/2 + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

