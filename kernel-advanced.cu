// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

const int BLOCK_SIZE = 512;

__device__ void warpReduce(volatile int* data, int threadId) {
    data[threadId] += data[threadId + 32];
    data[threadId] += data[threadId + 16];
    data[threadId] += data[threadId + 8];
    data[threadId] += data[threadId + 4];
    data[threadId] += data[threadId + 2];
    data[threadId] += data[threadId + 1];
}

__global__ void sumScanSingleBlock(int* changes, int* account, int clients) {
    int clientId = blockIdx.x;
    int threadId = threadIdx.x;
    int periods = blockDim.x * 2;

    int shared1Id = threadIdx.x * 2;
    int shared2Id = threadIdx.x * 2 + 1;

    int global1Id = clients * (shared1Id + blockDim.x * blockIdx.y * 2) + clientId;
    int global2Id = clients * (shared2Id + blockDim.x * blockIdx.y * 2) + clientId;

    extern __shared__ int shared[];
    // Load data into shared memory
    shared[shared1Id] = changes[global1Id];
    shared[shared2Id] = changes[global2Id];

    // Reduction phase
    int offset = 1;
    for (int d = periods >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (threadId == 0) {
        shared[periods - 1] = 0;
    }

    // Post-reduction phase
    for (int d = 1; d < periods; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            int t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }

    __syncthreads();
    // Write results to global memory
    shared[shared1Id] += changes[global1Id];
    shared[shared2Id] += changes[global2Id];
    account[global1Id] = shared[shared1Id];
    account[global2Id] = shared[shared2Id];
}

__global__ void sumScanMultiBlock(int* changes, int* account, int clients, int* temp) {
    int clientId = blockIdx.x;
    int threadId = threadIdx.x;
    int periods = blockDim.x * 2;

    int shared1Id = threadIdx.x * 2;
    int shared2Id = threadIdx.x * 2 + 1;

    int global1Id = clients * (shared1Id + blockDim.x * blockIdx.y * 2) + clientId;
    int global2Id = clients * (shared2Id + blockDim.x * blockIdx.y * 2) + clientId;

    // Block dim x - number of clients
    // Block dim y - 2

    extern __shared__ int shared[];
    // Load data into shared memory
    shared[shared1Id] = changes[global1Id];
    shared[shared2Id] = changes[global2Id];
    shared[shared1Id + blockDim.x * 2] = shared[shared1Id];
    shared[shared2Id + blockDim.x * 2] = shared[shared2Id];

    // Reduction sum
    for (int i = 1; i < blockDim.x * 2; i *= 2) {
        if (threadId % i == 0) {
            shared[threadId * 2 + blockDim.x * 2] += shared[threadId * 2 + i + blockDim.x * 2];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        for (int i = blockIdx.y + 1; i < gridDim.y; i++) {
            atomicAdd(&temp[gridDim.y * blockIdx.x + i], shared[blockDim.x * 2]);
        }
        // atomicExch(&atom[gridDim.y * blockIdx.x + blockIdx.y], 1);
    }

    // Reduction phase
    int offset = 1;
    for (int d = periods >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (threadId == 0) {
        shared[periods - 1] = 0;
    }

    // Post-reduction phase
    for (int d = 1; d < periods; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (threadId < d) {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            int t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }

    __syncthreads();
    // Write results to global memory
    shared[shared1Id] += changes[global1Id] + temp[gridDim.y * blockIdx.x + blockIdx.y];
    shared[shared2Id] += changes[global2Id] + temp[gridDim.y * blockIdx.x + blockIdx.y];
    account[global1Id] = shared[shared1Id];
    account[global2Id] = shared[shared2Id];
}

__global__ void sumReduceSingleBlock(int* account, int* sum, int clients, int periods) {
    extern __shared__ int shared[];
    // Each thread loads two elements into shared memory
    shared[threadIdx.x] = account[blockIdx.x * clients + threadIdx.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i >>= 1) {
        if (threadIdx.x < i) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        warpReduce(shared, threadIdx.x);
    }
    if (threadIdx.x == 0) {
        sum[blockIdx.x] = shared[0];
    }
}

// __global__ void sumReduceSingleBlock(int* account, int* sum, int clients, int periods) {
//     extern __shared__ int shared[];
//     // Each thread loads two elements into shared memory
//     shared[threadIdx.x * 2] =
//         account[blockIdx.x * clients + threadIdx.x * 2 + blockIdx.y * blockDim.x * 2];
//     shared[threadIdx.x * 2 + 1] =
//         account[blockIdx.x * clients + threadIdx.x * 2 + 1 + blockIdx.y * blockDim.x * 2];

//     for (int i = 1; i < blockDim.x * 2; i *= 2) {
//         if (threadIdx.x % i == 0) {
//             shared[threadIdx.x * 2] += shared[threadIdx.x * 2 + i];
//         }
//         __syncthreads();
//     }
//     if (threadIdx.x == 0) {
//         sum[blockIdx.x] = shared[0];
//     }
// }

// __global__ void kernel3(int* changes, int* account, int* sum, int clients, int* temp) {
//     int globalId = clients * (threadIdx.x + blockDim.x * blockIdx.y) + blockIdx.x;
//     account[globalId] += temp[gridDim.y * blockIdx.x + blockIdx.y];
// }

__global__ void sumkernel(int* account, int* sum, int clients, int periods, int* temp) {
    extern __shared__ int shared[];
    // Each thread loads two elements into shared memory
    shared[threadIdx.x * 2] =
        account[blockIdx.x * clients + threadIdx.x * 2 + blockIdx.y * blockDim.x * 2];
    shared[threadIdx.x * 2 + 1] =
        account[blockIdx.x * clients + threadIdx.x * 2 + 1 + blockIdx.y * blockDim.x * 2];

    for (int i = 1; i < blockDim.x * 2; i *= 2) {
        if (threadIdx.x % i == 0) {
            shared[threadIdx.x * 2] += shared[threadIdx.x * 2 + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        temp[blockIdx.x * gridDim.y + blockIdx.y] = shared[0];
    }
}

__global__ void sumkernel2(int* account, int* sum, int clients, int periods, int* temp) {
    extern __shared__ int shared[];
    shared[threadIdx.x * 2] = temp[blockIdx.x * 8 + threadIdx.x * 2];
    shared[threadIdx.x * 2 + 1] = temp[blockIdx.x * 8 + threadIdx.x * 2 + 1];

    for (int i = 1; i < blockDim.x * 2; i *= 2) {
        if (threadIdx.x % i == 0) {
            shared[threadIdx.x * 2] += shared[threadIdx.x * 2 + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sum[blockIdx.x] = shared[0];
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    //int* temp;
    //cudaMalloc((void**)&temp, sizeof(int) * clients * (periods / BLOCK_SIZE) / 2);
    /**
     * If periods is less than 2048, we can use kernel that calculates everything in one block
     * Otherwise we need to synchronize multiple blocks and that creates a lot of overhead
     */
    /*
    if (periods <= 2048) {
        int memory = sizeof(int) * periods;
        sumScanSingleBlock<<<clients, periods / 2, memory>>>(changes, account, clients);
    } else {
        int memory = sizeof(int) * (BLOCK_SIZE * 4);
        dim3 grid(clients, (periods / BLOCK_SIZE) / 2);
        sumScanMultiBlock<<<grid, BLOCK_SIZE, memory>>>(changes, account, clients, temp);
    }

    {
        int memory = sizeof(int) * clients;
        sumReduceSingleBlock<<<periods, clients, memory>>>(account, sum, clients, periods);
    }*/
    // dim3 grid(clients, (periods / BLOCK_SIZE) / 2);
    // dim3 block(BLOCK_SIZE);
    // kernel2<<<grid, block, sizeof(int) * (BLOCK_SIZE * 4)>>>(changes, account, sum, clients,
    // temp); kernel3<<<grid, BLOCK_SIZE * 2>>>(changes, account, sum, clients, temp);
    // dim3 sumblocks(periods, (clients / BLOCK_SIZE) / 2);
    // sumkernel<<<sumblocks, BLOCK_SIZE, sizeof(int) * (BLOCK_SIZE * 2)>>>(account, sum, clients,
    //                                                                      periods, temp);
    // int threads = (clients / BLOCK_SIZE) / 4;
    // // printf("Threads: %d\n", threads);
    // sumkernel2<<<periods, threads, sizeof(int) * threads * 2>>>(account, sum, clients, periods,
    //                                                             temp);
    kernel<<<clients, 1>>>(changes, account, sum, clients, periods);
}
