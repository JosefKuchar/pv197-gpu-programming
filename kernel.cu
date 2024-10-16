// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

// __global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     account[i] = changes[i];
//     atomicAdd(&sum[0], account[i]);
//     for (int j = 1; j < periods; j++) {
//         int test = account[j * clients + i] =
//             account[(j - 1) * clients + i] + changes[j * clients + i];
//         atomicAdd(&sum[j], account[j * clients + i]);
//     }
// }

#define COLS 16
#define ROWS 64
__device__ int temp[8192 * ROWS];

// __global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int initChange = changes[i + threadIdx.y * clients * periods / ROWS];
//     account[i + threadIdx.y * clients * periods / ROWS] = initChange;
//     // atomicAdd(&sum[0], account[i]);
//     int s = initChange;
//     for (int j = 1; j < periods / ROWS; j++) {
//         int jj = j + threadIdx.y * periods / ROWS;
//         int change = changes[jj * clients + i];
//         s += change;
//         account[jj * clients + i] = account[(jj - 1) * clients + i] + change;
//         // atomicAdd(&sum[j], account[j * clients + i]);
//     }

//     for (int j = threadIdx.y + 1; j < ROWS; j++) {
//         atomicAdd(&temp[i * ROWS + j], s);
//     }
// }

// __global__ void kernel2(int* account, int* sum, int clients, int periods) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int j = 0; j < periods / ROWS; j++) {
//         int jj = j + threadIdx.y * periods / ROWS;
//         account[jj * clients + i] += temp[i * ROWS + threadIdx.y];
//         // atomicAdd(&sum[jj], account[jj * clients + i]);
//     }
// }

__global__ void sumScanMultiBlock(int* changes, int* account, int clients, int periods) {
    extern __shared__ int shared[];

    int p = (periods / ROWS);
    // Offset in shared memory for original values
    int originalOffset = ROWS * COLS * 4;
    // Sum offset for reduction sum for shared memory
    int sumOffset = ROWS * COLS * 2;
    // "General" block offset for shared memory
    int blockOffset = threadIdx.x * ROWS * 2;
    // Indices for shared memory
    int shared1Id = blockOffset + threadIdx.y * 2;
    int shared2Id = blockOffset + threadIdx.y * 2 + 1;
    // Indices for global memory
    int global1Id = blockIdx.x * blockDim.x + threadIdx.x +
                    clients * (blockIdx.y * blockDim.y + threadIdx.y * 2);
    int global2Id = blockIdx.x * blockDim.x + threadIdx.x +
                    clients * (blockIdx.y * blockDim.y + threadIdx.y * 2 + 1);
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
    for (int i = 1; i < ROWS * 2; i *= 2) {
        if (threadIdx.y % i == 0) {
            int offset = blockOffset + sumOffset;
            shared[offset + threadIdx.y * 2] += shared[offset + threadIdx.y * 2 + i];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        for (int i = blockIdx.y + 1; i < gridDim.y; i++) {
            atomicAdd(&temp[i + gridDim.y * (blockIdx.x * blockDim.x + threadIdx.x)],
                      shared[blockOffset + sumOffset]);
        }
    }

    // Reduction phase
    int offset = 1;
    for (int d = p >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadIdx.y < d) {
            int ai = offset * (2 * threadIdx.y + 1) - 1;
            int bi = offset * (2 * threadIdx.y + 2) - 1;
            shared[blockOffset + bi] += shared[blockOffset + ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (threadIdx.y == 0) {
        shared[blockOffset + p - 1] = 0;
    }

    // Post-reduction phase
    for (int d = 1; d < p; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadIdx.y < d) {
            int ai = offset * (2 * threadIdx.y + 1) - 1;
            int bi = offset * (2 * threadIdx.y + 2) - 1;
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

__global__ void kernel2(int* account, int clients, int periods) {
    // Indices for global memory
    int global1Id = blockIdx.x * blockDim.x + threadIdx.x +
                    clients * (blockIdx.y * blockDim.y + threadIdx.y * 2);
    int global2Id = blockIdx.x * blockDim.x + threadIdx.x +
                    clients * (blockIdx.y * blockDim.y + threadIdx.y * 2 + 1);
    account[global1Id] += temp[blockIdx.y + gridDim.y * (blockIdx.x * blockDim.x + threadIdx.x)];
    account[global2Id] += temp[blockIdx.y + gridDim.y * (blockIdx.x * blockDim.x + threadIdx.x)];
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    // 2 (2 values per thread) * 2 (each value is duplicated twice)
    int memory = sizeof(int) * COLS * ROWS * 2 * 4;
    dim3 grid(COLS, ROWS / 2);  // Each thread processes two rows of client
    dim3 block(COLS, ROWS);
    sumScanMultiBlock<<<grid, block, memory>>>(changes, account, clients, periods);
    kernel2<<<grid, block>>>(account, clients, periods);
}
