#define BLOCK_SIZE 128
#define P 16

__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache[P];
    unsigned sums[P];
    for (int j = 0; j < 8192 / P; j++) {
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
            int warp_sum = cache[k];
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
            sums[k] = warp_sum;
        }
#pragma unroll
        for (int k = 0; k < P; k++) {
            account[index + k * clients] = cache[k];
        }
        if (threadIdx.x % 32 == 0) {
#pragma unroll
            for (int k = 0; k < P; k++) {
                atomicAdd(&sum[j * P + k], sums[k]);
            }
        }
        index += P * 8192;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
