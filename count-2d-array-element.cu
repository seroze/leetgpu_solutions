#include <cuda_runtime.h>


// __global__ void count_kernel(const int* input, int* output, int N, int M, int K){

//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int col = blockDim.x * blockIdx.x + threadIdx.x;   
//     __shared__ int block_count;//shared per block 


//     if (row<N && col<M){
        

//         if (threadIdx.x == 0 && threadIdx.y == 0)
//             block_count = 0;

//         __syncthreads(); // wait for all of them to finish

//         int local = 0; // local variable, local to thread  
//         if (input[M*row +col] == K) {
//             local = 1; 
//         }

//         if (local)
//             atomicAdd(&block_count, local);

//         __syncthreads(); // wait for everyone to finish

//         if (threadIdx.x == 0)
//             atomicAdd(output, block_count);
//     }
// }

__global__ void count_kernel(
    const int* input,
    int* output,
    int N,
    int M,
    int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int block_count;

    if (threadIdx.x == 0 && threadIdx.y == 0)
        block_count = 0;

    __syncthreads();

    if (row < N && col < M) {
        if (input[row * M + col] == K)
            atomicAdd(&block_count, 1);
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
        atomicAdd(output, block_count);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {

    cudaMemset(output, 0, sizeof(int));

    int TILE_WIDTH = 16;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+TILE_WIDTH-1)/ TILE_WIDTH, (M+TILE_WIDTH-1)/TILE_WIDTH);

    count_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
}
