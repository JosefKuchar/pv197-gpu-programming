/**
 * @brief CUDA kernel for prefix sum and reduction
 * Uses naive prefix sum and warp shuffle reduction
 * Optimized for memory access
 * Special case for 8192x8192 matrix for better compiler optimization
 *
 * @author Josef Kuchař (567769)
 */

// Number of columns in block
#define BLOCK_SIZE 128
// Number of rows to preload in each iteration
#define PRELOAD_COUNT 16

#define ROWS 4

/**
 * @brief General kernel
 *
 * @param changes (Input) Account changes
 * @param account (Output) Account balances
 * @param sum (Output) Period sums
 * @param clients Client count
 * @param periods Period count
 */
__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    __shared__ volatile int shared[BLOCK_SIZE * ROWS];

    int index = blockIdx.x * blockDim.x + threadIdx.x + threadIdx.y * clients * PRELOAD_COUNT;

    int acc = 0;
    int cache[PRELOAD_COUNT];
    unsigned sums[PRELOAD_COUNT];
    for (int j = 0; j < periods / (PRELOAD_COUNT * ROWS); j++) {
        // Load PRELOAD_COUNT rows from global memory
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            cache[k] = changes[index + k * clients];
        }

        // Calculate prefix sum
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            acc += cache[k];
            int warp_sum = acc;
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
            cache[k] = acc;
            sums[k] = warp_sum;
        }

        // Store PRELOAD_COUNT rows to global memory
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            account[index + k * clients] = cache[k];
        }

        // First thread in warp stores the reduction sum
        if (threadIdx.x % 32 == 0) {
#pragma unroll
            for (int k = 0; k < PRELOAD_COUNT; k++) {
                atomicAdd(&sum[j * PRELOAD_COUNT + k], sums[k]);
            }
        }
        index += ROWS * PRELOAD_COUNT * clients;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    dim3 block(BLOCK_SIZE, ROWS);

    // General "slower" kernel
    kernel<<<clients / BLOCK_SIZE, block>>>(changes, account, sum, clients, periods);
}
