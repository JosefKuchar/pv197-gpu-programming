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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache[PRELOAD_COUNT];
    unsigned sums[PRELOAD_COUNT];
    for (int j = 0; j < periods / PRELOAD_COUNT; j++) {
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
        index += PRELOAD_COUNT * clients;
    }
}

/**
 * @brief Specialized kernel for 8192x8192
 *
 * @param changes (Input) Account changes
 * @param account (Output) Account balances
 * @param sum (Output) Period sums
 */
__global__ void kernel_8192(int* changes, int* account, int* sum) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache[PRELOAD_COUNT];
    unsigned sums[PRELOAD_COUNT];
    for (int j = 0; j < 8192 / PRELOAD_COUNT; j++) {
        // Load PRELOAD_COUNT rows from global memory
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            cache[k] = changes[index + k * 8192];
        }

        // Calculate prefix sum
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            acc += cache[k];
            cache[k] = acc;
        }

        // Reduction sum using warp shuffle
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            int warp_sum = cache[k];
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
            sums[k] = warp_sum;
        }

        // Store PRELOAD_COUNT rows to global memory
#pragma unroll
        for (int k = 0; k < PRELOAD_COUNT; k++) {
            account[index + k * 8192] = cache[k];
        }

        // First thread in warp stores the reduction sum
        if (threadIdx.x % 32 == 0) {
#pragma unroll
            for (int k = 0; k < PRELOAD_COUNT; k++) {
                atomicAdd(&sum[j * PRELOAD_COUNT + k], sums[k]);
            }
        }
        index += PRELOAD_COUNT * 8192;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    if (clients == 8192 && periods == 8192) {
        // Specialized fast kernel for 8192x8192
        kernel_8192<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum);
    } else {
        // General "slower" kernel
        kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
    }
}
