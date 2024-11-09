// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://web.archive.org/web/20110924131401/http://www.moderngpu.com/intro/scan.html

#define COLS 16
#define ROWS 32
// __device__ int temp[32 * 8192];
__global__ void firstStage(int* changes, int* account, int* sum, int clients, int periods) {
    __shared__ int temp[COLS * ROWS];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index = blockIdx.x * COLS + tx + clients * (ty) * 256;
    int val = 0;
    int temp_account[256];
#pragma unroll 4
    for (int i = 0; i < 256; i++) {
        val += changes[index];
        temp_account[i] = val;
        index += clients;
    }

    // Store to temp
    temp[tx + ty * COLS] = val;
    __syncthreads();

    // Exclusive scan in shared memory
    // TODO: Do with multiple threads
    if (ty == 0) {
        int sum = 0;
        for (int i = 0; i < 32; i++) {
            int current = temp[tx + i * COLS];
            temp[tx + i * COLS] = sum;
            sum += current;
        }
    }
    __syncthreads();
    // Store back to global memory
    index = blockIdx.x * COLS + tx + clients * (ty) * 256;
    // printf("tx: %d, ty: %d, i: %d, v: %d\n", tx, ty, tx + ty * COLS, temp[tx + ty * COLS]);
    int t = temp[tx + ty * COLS];
#pragma unroll 4
    for (int i = 0; i < 256; i++) {
        temp_account[i] += t;
        account[index] = temp_account[i];
        // atomicAdd(&sum[i + ty * 256], temp_account[i]);
        index += clients;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 4);

    dim3 block(COLS, ROWS);
    firstStage<<<clients / COLS, block>>>(changes, account, sum, clients, periods);

    // Output memory errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}
