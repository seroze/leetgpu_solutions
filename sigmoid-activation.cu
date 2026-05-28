#include <cuda_runtime.h>
#include <math.h>

__device__  double sigmoid(double x) {
    return 1.0 / (1.0 + expf(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 

    if (tid<N) {
        Y[tid] = sigmoid(X[tid]);
    }

}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}

