// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://web.archive.org/web/20110924131401/http://www.moderngpu.com/intro/scan.html

#define WARP_SIZE 32
// ROWS = 32 => 6 iterations
// ROWS = 16 => 5 iterations
// log2(ROWS) + 1 iterations for reduction phase

#define COLS 128
#define ROWS 256
__device__ int temp[32 * 8192];
__global__ void firstStage(int* changes, int* account, int clients, int periods) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + clients * blockIdx.y * ROWS;
    int val = 0;
#pragma unroll
    for (int i = 0; i < ROWS; i++) {
        val += changes[index];
        account[index] = val;
        index += clients;
    }
    // Store sum for next stage
    temp[blockIdx.y + (blockIdx.x * blockDim.x + threadIdx.x) * 32] = val;
}

__global__ void secondStage() {
    __shared__ volatile int scan[WARP_SIZE];
    int tid = threadIdx.x;

    // Read from global memory.
    int x = temp[threadIdx.x + blockIdx.x * blockDim.x];
    scan[tid] = x;

    // Run each pass of the scan.
    int sum = x;
// #pragma unroll
// for(int offset = 1; offset < WARP_SIZE; offset *= 2) {
// This code generates
//  * Advisory: Loop was not unrolled, cannot deduce loop trip count"
// We want to use the above iterators, but nvcc totally sucks. It only
// unrolls loops when the conditional is simply incremented.
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        // Counting from i = 0 to 5 and shifting 1 by that number of bits
        // generates the desired offset sequence of (1, 2, 4, 8, 16).
        int offset = 1 << i;

        // Add tid - offset into sum, if this does not put us past the beginning
        // of the array. Write the sum back into scan array.
        if (tid >= offset)
            sum += scan[tid - offset];
        scan[tid] = sum;
    }
    temp[threadIdx.x + blockIdx.x * blockDim.x] = sum - x;
}

__global__ void thirdStage(int* account, int clients) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + clients * blockIdx.y * ROWS;
    int add = temp[blockIdx.y + (blockIdx.x * blockDim.x + threadIdx.x) * 32];

#pragma unroll
    for (int i = 0; i < ROWS; i++) {
        account[index] += add;
        index += clients;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    dim3 grid(clients / COLS, periods / ROWS);
    firstStage<<<grid, COLS>>>(changes, account, clients, periods);
    secondStage<<<clients, 32>>>();
    thirdStage<<<grid, COLS>>>(account, clients);
    // sumScan2<<<clients, grid.y / 2, sizeof(int) * grid.y>>>(grid.y);
    // kernel2<<<grid, block>>>(account, clients, periods);

    // Output memory errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
}
