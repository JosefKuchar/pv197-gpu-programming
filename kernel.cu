/**
 * @brief CUDA kernel for prefix sum and reduction
 * Uses naive prefix sum and warp shuffle reduction
 * Optimized for memory access
 * Special case for 8192x8192 matrix for better compiler optimization
 *
 * @author Josef Kuchař (567769)
 */

// Number of columns in block
#define BLOCK_SIZE 64
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
    __shared__ volatile int shared[BLOCK_SIZE * PRELOAD_COUNT * 2];

    int cache[PRELOAD_COUNT];
    int sums[PRELOAD_COUNT];
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Producer warps
    if (threadIdx.y == 0) {
        bool offset_flag = false;
        for (int i = 0; i < periods / PRELOAD_COUNT; i++) {
            int offset = offset_flag ? BLOCK_SIZE * PRELOAD_COUNT : 0;

            // Load PRELOAD_COUNT rows from global memory
#pragma unroll
            for (int j = 0; j < PRELOAD_COUNT; j++) {
                cache[j] = changes[index + j * clients];
            }
            // Store PRELOAD_COUNT rows to shared memory
#pragma unroll
            for (int j = 0; j < PRELOAD_COUNT; j++) {
                shared[offset + threadIdx.x + j * blockDim.x] = cache[j];
            }

            index += clients * PRELOAD_COUNT;
            offset_flag = !offset_flag;

            __syncthreads();
        }
    }

    __syncthreads();

    // Consumer warps
    int acc = 0;
    if (threadIdx.y == 1) {
        bool offset_flag = false;
        for (int i = 0; i < periods / PRELOAD_COUNT; i++) {
            int offset = offset_flag ? BLOCK_SIZE * PRELOAD_COUNT : 0;
            // Load PRELOAD_COUNT rows from shared memory
#pragma unroll
            for (int j = 0; j < PRELOAD_COUNT; j++) {
                cache[j] = shared[offset + threadIdx.x + j * blockDim.x];
            }

            // Calculate prefix sum
#pragma unroll
            for (int j = 0; j < PRELOAD_COUNT; j++) {
                acc += cache[j];
                int warp_sum = acc;
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
                cache[j] = acc;
                sums[j] = warp_sum;
            }

            // Store to global
#pragma unroll
            for (int j = 0; j < PRELOAD_COUNT; j++) {
                account[index + j * clients] = cache[j];
            }

            // Store sums
            if (threadIdx.x % 32 == 0) {
#pragma unroll
                for (int j = 0; j < PRELOAD_COUNT; j++) {
                    atomicAdd(&sum[i * PRELOAD_COUNT + j], sums[j]);
                }
            }

            index += clients * PRELOAD_COUNT;
            offset_flag = !offset_flag;

            __syncthreads();
        }
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    dim3 block(BLOCK_SIZE, 2);

    // General "slower" kernel
    kernel<<<clients / BLOCK_SIZE, block>>>(changes, account, sum, clients, periods);
}
