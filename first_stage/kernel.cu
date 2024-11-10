const int BLOCK_SIZE = 32;

#define P 8

__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    __shared__ volatile int shared[BLOCK_SIZE];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache[P];
    for (int j = 0; j < periods / P; j++) {
#pragma unroll
        for (int k = 0; k < P; k++) {
            cache[k] = changes[index + k * clients];
        }
#pragma unroll
        for (int k = 0; k < P; k++) {
            acc += cache[k];
            cache[k] = acc;
        }
#pragma unroll
        for (int k = 0; k < P; k++) {
            int idx = threadIdx.x + k * BLOCK_SIZE;
            shared[idx] = cache[k];
            if (threadIdx.x < 16) {
                shared[idx] += shared[idx + 16];
                shared[idx] += shared[idx + 8];
                shared[idx] += shared[idx + 4];
                shared[idx] += shared[idx + 2];
                shared[idx] += shared[idx + 1];
            }
            if (threadIdx.x == 0) {
                atomicAdd(&sum[j * P + k], shared[idx]);
            }
        }
#pragma unroll
        for (int k = 0; k < P; k++) {
            account[index + k * clients] = cache[k];
        }
        index += P * clients;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
