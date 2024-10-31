// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#define COLS 64
#define ROWS 8
__device__ int blockSumStorage[8192 * (8192 / ROWS) / 2];
__global__ void sumScanMultiBlock(int* changes, int* account, int clients, int periods) {
    extern __shared__ int shared[];
    int localCol = threadIdx.y;
    int localRow = threadIdx.x;
    int localColDim = blockDim.y;
    int localRowDim = blockDim.x;
    int gridCol = blockIdx.x;
    int gridRow = blockIdx.y;
    int gridRowDim = gridDim.y;

    int p = ROWS * 2;
    // Offset in shared memory for original values
    int originalOffset = ROWS * COLS * 4;
    // Sum offset for reduction sum for shared memory
    int sumOffset = ROWS * COLS * 2;
    // "General" block offset for shared memory
    int blockOffset = localCol * ROWS * 2;
    // Indices for shared memory
    int shared1Id = blockOffset + localRow * 2;
    int shared2Id = blockOffset + localRow * 2 + 1;
    // Indices for global memory
    int global1Id =
        gridCol * localColDim + localCol + clients * (gridRow * localRowDim * 2 + localRow * 2);
    int global2Id =
        gridCol * localColDim + localCol + clients * (gridRow * localRowDim * 2 + localRow * 2 + 1);
    // Load data into shared memory
    shared[shared1Id] = changes[global1Id];
    shared[shared2Id] = changes[global2Id];
    // Duplicate values for reduction sum
    shared[shared1Id + sumOffset] = shared[shared1Id];
    shared[shared2Id + sumOffset] = shared[shared2Id];
    // Save original values for post-reduction phase
    shared[shared1Id + originalOffset] = shared[shared1Id];
    shared[shared2Id + originalOffset] = shared[shared2Id];

    // Reduction sum
#pragma unroll
    for (int i = 1; i < ROWS * 2; i *= 2) {
        if (localRow % i == 0) {
            int offset = blockOffset + sumOffset;
            shared[offset + localRow * 2] += shared[offset + localRow * 2 + i];
        }
        __syncthreads();
    }

    if (localRow == 0) {
        blockSumStorage[gridRow + gridRowDim * (gridCol * localColDim + localCol)] =
            shared[blockOffset + sumOffset];
    }

    // Reduction phase
    int offset = 1;
#pragma unroll
    for (int d = p >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (localRow < d) {
            int ai = offset * (2 * localRow + 1) - 1;
            int bi = offset * (2 * localRow + 2) - 1;
            shared[blockOffset + bi] += shared[blockOffset + ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (localRow == 0) {
        shared[blockOffset + p - 1] = 0;
    }

    // Post-reduction phase
#pragma unroll
    for (int d = 1; d < p; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (localRow < d) {
            int ai = offset * (2 * localRow + 1) - 1;
            int bi = offset * (2 * localRow + 2) - 1;
            int t = shared[blockOffset + ai];
            shared[blockOffset + ai] = shared[blockOffset + bi];
            shared[blockOffset + bi] += t;
        }
    }

    __syncthreads();
    // Add original values to the results, because algorithm produces prefix sum without them
    shared[shared1Id] += shared[shared1Id + originalOffset];
    shared[shared2Id] += shared[shared2Id + originalOffset];
    // Write results back to global memory
    account[global1Id] = shared[shared1Id];
    account[global2Id] = shared[shared2Id];
}

__global__ void sumScan2(int rowDim) {
    extern __shared__ int shared[];
    int shared1Id = threadIdx.x * 2;
    int shared2Id = threadIdx.x * 2 + 1;
    int global1Id = threadIdx.x * 2 + rowDim * blockIdx.x;
    int global2Id = threadIdx.x * 2 + 1 + rowDim * blockIdx.x;
    shared[shared1Id] = blockSumStorage[global1Id];
    shared[shared2Id] = blockSumStorage[global2Id];
    int p = rowDim;

    // Reduction phase
    int offset = 1;
#pragma unroll
    for (int d = p >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (threadIdx.x == 0) {
        shared[p - 1] = 0;
    }

    // Post-reduction phase
#pragma unroll
    for (int d = 1; d < p; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < d) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            int t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }

    __syncthreads();
    // Write results back to global memory
    blockSumStorage[global1Id] = shared[shared1Id];
    blockSumStorage[global2Id] = shared[shared2Id];
}
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
    // 2 (2 values per thread) * 2 (each value is duplicated twice)
    int memory = sizeof(int) * COLS * ROWS * 2 * 3;
    dim3 grid(clients / COLS,
              (periods / ROWS) / 2);  // Each thread processes two rows of client
    dim3 block(ROWS, COLS);
    sumScanMultiBlock<<<grid, block, memory>>>(changes, account, clients, periods);
    sumScan2<<<clients, grid.y / 2, sizeof(int) * grid.y>>>(grid.y);
    // kernel2<<<grid, block>>>(account, clients, periods);
}
