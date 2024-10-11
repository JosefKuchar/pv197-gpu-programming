// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    account[i] = changes[i];
    atomicAdd(&sum[0], account[i]);
    for (int j = 1; j < periods; j++) {
        account[j * clients + i] = account[(j - 1) * clients + i] + changes[j * clients + i];
        atomicAdd(&sum[j], account[j * clients + i]);
    }
}

__global__ void kernel2(int* changes, int* account, int* sum, int clients, int periods) {
    int clientId = blockIdx.x;
    int threadId = threadIdx.x;
    extern __shared__ int shared[];
    // Load data into shared memory
    shared[threadId * 2] = changes[clients * threadId * 2 + clientId];
    shared[threadId * 2 + 1] = changes[clients * (threadId * 2 + 1) + clientId];

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

    // Write results to global memory
    __syncthreads();
    shared[threadId * 2] += changes[clients * threadId * 2 + clientId];
    shared[threadId * 2 + 1] += changes[clients * (threadId * 2 + 1) + clientId];
    atomicAdd(&sum[threadId * 2], shared[threadId * 2]);
    atomicAdd(&sum[threadId * 2 + 1], shared[threadId * 2 + 1]);
    account[clients * threadId * 2 + clientId] = shared[threadId * 2];
    account[clients * (threadId * 2 + 1) + clientId] = shared[threadId * 2 + 1];
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel2<<<clients, periods / 2, sizeof(int) * periods>>>(changes, account, sum, clients,
                                                             periods);
    // kernel<<<clients, 1>>>(changes, account, sum, clients, periods);
}
