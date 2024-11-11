#define BLOCK_SIZE 64
#define P 16

__global__ void kernel(int* changes, int* account, int* sum) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache[P];
    for (int j = 0; j < 8192 / P; j++) {
        int offset = index;
#pragma unroll
        for (int k = 0; k < P; k++) {
            cache[k] = changes[offset];
            offset += 8192;
        }
#pragma unroll
        for (int k = 0; k < P; k++) {
            acc += cache[k];
            cache[k] = acc;
            int warp_sum = acc;
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
            if (threadIdx.x % 32 == 0) {
                atomicAdd(&sum[j * P + k], warp_sum);
            }
        }
        offset = index;
#pragma unroll
        for (int k = 0; k < P; k++) {
            account[offset] = cache[k];
            offset += 8192;
        }
        index += P * 8192;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum);
}
