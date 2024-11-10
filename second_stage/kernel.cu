// TODO: Use advanced techniques from
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://web.archive.org/web/20110924131401/http://www.moderngpu.com/intro/scan.html

#define BLOCK_COLS 32
#define COLS 8
#define ROWS 4

// __global__ void firstStage(int* changes, int* account, int* sum, int clients, int periods) {
//     __shared__ volatile int temp[BLOCK_COLS * ROWS * 2];

//     int tx = threadIdx.x % COLS;
//     int ty = (threadIdx.x / COLS) % ROWS;
//     int xx = threadIdx.x / (COLS * ROWS);

//     int index = tx + xx * COLS + blockIdx.x * BLOCK_COLS + clients * ty;

//     int sharedIndex = ty + tx * (ROWS + ROWS / 2) + xx * (ROWS * COLS * 2);
//     int prev = 0;
//     for (int i = 0; i < 2048; i++) {
//         temp[sharedIndex] = changes[index] + prev;
//         temp[sharedIndex + 1] += temp[sharedIndex];
//         temp[sharedIndex + 2] += temp[sharedIndex];
//         if (ty == 0) {
//             prev = temp[sharedIndex + 3];
//         }
//         account[index] = temp[sharedIndex];
//         // atomicAdd(&sum[4 * i + ty], temp[sharedIndex]);
//         index += clients * ROWS;
//     }
// }

__global__ void firstStage(int* changes, int* account, int* sum, int clients, int periods) {
    __shared__ volatile int temp[BLOCK_COLS * ROWS];

    int tx = threadIdx.x % COLS;
    int ty = (threadIdx.x / COLS) % ROWS;
    int xx = threadIdx.x / (COLS * ROWS);

    int index = tx + xx * COLS + blockIdx.x * BLOCK_COLS + clients * ty;

    int sharedIndex = ty + tx * ROWS + xx * (ROWS * COLS);
    int prev = 0;
    for (int i = 0; i < 2048; i++) {
        temp[sharedIndex] = changes[index] + prev;
        if (ty < 3) {
            temp[sharedIndex + 1] += temp[sharedIndex];
        }
        if (ty < 2) {
            temp[sharedIndex + 2] += temp[sharedIndex];
        }
        if (ty == 0) {
            prev = temp[sharedIndex + 3];
        }
        account[index] = temp[sharedIndex];
        temp[sharedIndex] += temp[sharedIndex + 4 * 4];
        temp[sharedIndex] += temp[sharedIndex + 4 * 2];
        temp[sharedIndex] += temp[sharedIndex + 4];
        if (tx == 0) {
            atomicAdd(&sum[4 * i + ty], temp[sharedIndex]);
        }
        index += clients * ROWS;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    firstStage<<<clients / BLOCK_COLS, BLOCK_COLS * ROWS>>>(changes, account, sum, clients,
                                                            periods);

    // Output memory errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}
