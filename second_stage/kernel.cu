// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://web.archive.org/web/20110924131401/http://www.moderngpu.com/intro/scan.html

#define COLS 32
#define ROWS 32
// __device__ int temp[32 * 8192];
__global__ void firstStage(int* changes, int* account, int* sum, int clients, int periods) {
    __shared__ int temp[COLS * ROWS];
    int index = blockIdx.x * 32 + threadIdx.x % COLS + clients * (threadIdx.x / COLS) * 256;
    int val = 0;
#pragma unroll
    for (int i = 0; i < 256; i++) {
        val += changes[index];
        account[index] = val;
        index += clients;
    }

    // // Store to temp
    // temp[threadIdx.x + threadIdx.y * 32] = val;
    // __syncthreads();

    // // Exclusive scan in shared memory
    // // TODO: Do with multiple threads
    // if (threadIdx.y == 0) {
    //     int sum = 0;
    //     for (int i = 0; i < 32; i++) {
    //         int current = temp[threadIdx.x + i * 32];
    //         temp[threadIdx.x + i * 32] = sum;
    //         sum += current;
    //     }
    // }
    // __syncthreads();
    // // Store back to global memory
    // index = blockIdx.x * blockDim.x + threadIdx.x + clients * threadIdx.y * 256;
    // for (int i = 0; i < 256; i++) {
    //     account[index] += temp[threadIdx.x + threadIdx.y * 32];
    //     index += clients;
    // }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    firstStage<<<clients / COLS, COLS * ROWS>>>(changes, account, sum, clients, periods);

    // Output memory errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
}
