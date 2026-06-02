#include <cuda_runtime.h>


__global__ void reduce(const float* input, float* output, int N) {

    // use stride logic 
    int tid = threadIdx.x; 
    int idx = blockDim.x * blockIdx.x + tid ;
    __shared__ float sdata[256];

    sdata[tid]  = (idx<N) ? input[idx]: 0.0f; 
    __syncthreads();
   
    for(int stride = blockDim.x/2; stride > 0; stride/=2){
        if (tid<stride){
            sdata[tid] += sdata[tid+stride];
        }
            __syncthreads();
    }

    // write block result
    // only thread 0 of each block should perform this operation  
    if (tid == 0){
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {


    const int THREADS_PER_BLOCK = 256;
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
    dim3 blocksPerGrid((N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK);
    // ensure output is set to 0 initially 
    cudaMemset(output, 0, sizeof(float));
    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
