#include <cuda_runtime.h>
#include <math.h> 
#include <math_functions.h>

__device__ __forceinline__ float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}


__global__ void swiglu_kernel(const float* input, float* output, int halfN) {


    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx < halfN) {
        float leftHalf = input[idx];
        float rightHalf = input[idx+halfN]; 
        float res = leftHalf * sigmoid(leftHalf) * rightHalf;
        output[idx] = res;  
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
