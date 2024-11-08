// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://web.archive.org/web/20110924131401/http://www.moderngpu.com/intro/scan.html

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// ROWS = 32 => 6 iterations
// ROWS = 16 => 5 iterations
// log2(ROWS) + 1 iterations for reduction phase

#define COLS 32
#define ROWS 32
// __device__ int blockSumStorage[8192 * (8192 / ROWS) / 2];
__global__ void sumScanMultiBlock(int* changes, int* account, int clients, int periods) {
    __shared__ volatile int scan[ROWS * COLS];
    int localCol = threadIdx.y;
    int localRow = threadIdx.x;
    int localColDim = blockDim.y;
    int localRowDim = blockDim.x;
    int gridCol = blockIdx.x;
    int gridRow = blockIdx.y;
    int gridRowDim = gridDim.y;

    int blockOffset = localCol * ROWS;
    int globalId = gridCol * localColDim + localCol + clients * (gridRow * localRowDim + localRow);

    int x = changes[globalId];
    scan[localRow + blockOffset] = x;
    int sum = x;

#pragma unroll
    for (int i = 0; i < 5; ++i) {
        int offset = 1 << i;
        if (localRow >= offset)
            sum += scan[localRow + blockOffset - offset];
        scan[localRow + blockOffset] = sum;
    }

    // Write results back to global memory
    account[globalId] = sum;
}

// __global__ void sumScan2(int rowDim) {
//     extern __shared__ int shared[];
//     int shared1Id = threadIdx.x * 2;
//     int shared2Id = threadIdx.x * 2 + 1;
//     int global1Id = threadIdx.x * 2 + rowDim * blockIdx.x;
//     int global2Id = threadIdx.x * 2 + 1 + rowDim * blockIdx.x;
//     shared[shared1Id] = blockSumStorage[global1Id];
//     shared[shared2Id] = blockSumStorage[global2Id];
//     int p = rowDim;

//     // Reduction phase
//     int offset = 1;
// #pragma unroll
//     for (int d = p >> 1; d > 0; d >>= 1) {
//         __syncthreads();
//         if (threadIdx.x < d) {
//             int ai = offset * (2 * threadIdx.x + 1) - 1;
//             int bi = offset * (2 * threadIdx.x + 2) - 1;
//             shared[bi] += shared[ai];
//         }
//         offset *= 2;
//     }

//     // Clear the last element
//     if (threadIdx.x == 0) {
//         shared[p - 1] = 0;
//     }

//     // Post-reduction phase
// #pragma unroll
//     for (int d = 1; d < p; d *= 2) {
//         offset >>= 1;
//         __syncthreads();
//         if (threadIdx.x < d) {
//             int ai = offset * (2 * threadIdx.x + 1) - 1;
//             int bi = offset * (2 * threadIdx.x + 2) - 1;
//             int t = shared[ai];
//             shared[ai] = shared[bi];
//             shared[bi] += t;
//         }
//     }

//     __syncthreads();
//     // Write results back to global memory
//     blockSumStorage[global1Id] = shared[shared1Id];
//     blockSumStorage[global2Id] = shared[shared2Id];
// }
// __global__ void kernel2(int* account, int clients, int periods) {
//     int localCol = threadIdx.x;
//     int localRow = threadIdx.y;
//     // Indices for global memory
//     int global1Id =
//         gridCol * localColDim + localCol + clients * (gridRow * localRowDim + localRow *
//         2);
//     int global2Id =
//         gridCol * localColDim + localCol + clients * (gridRow * localRowDim + localRow * 2
//         + 1);
//     account[global1Id] += temp[gridRow + gridRowDim * (gridCol * localColDim + localCol)];
//     account[global2Id] += temp[gridRow + gridRowDim * (gridCol * localColDim + localCol)];
// }

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    dim3 grid(clients / COLS, periods / ROWS);
    dim3 block(ROWS, COLS);
    sumScanMultiBlock<<<grid, block>>>(changes, account, clients, periods);
    // sumScan2<<<clients, grid.y / 2, sizeof(int) * grid.y>>>(grid.y);
    // kernel2<<<grid, block>>>(account, clients, periods);
}
