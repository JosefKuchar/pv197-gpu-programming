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
    temp[blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * 32] = val;
}

__global__ void secondStage() {
    __shared__ volatile int scan[WARP_SIZE + WARP_SIZE / 2];
    volatile int* s = scan + WARP_SIZE / 2 + threadIdx.x;
    int x = temp[threadIdx.x + blockIdx.x * blockDim.x];
    s[0] = x;
    int sum = x;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        int offset = 1 << i;
        int y = s[-offset];
        sum += y;
        s[0] = sum;
    }
    temp[threadIdx.x + blockIdx.x * blockDim.x] = sum - x;
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    dim3 grid(clients / COLS, periods / ROWS);
    firstStage<<<grid, COLS>>>(changes, account, clients, periods);
    secondStage<<<clients, 32>>>();
    // sumScan2<<<clients, grid.y / 2, sizeof(int) * grid.y>>>(grid.y);
    // kernel2<<<grid, block>>>(account, clients, periods);
}
