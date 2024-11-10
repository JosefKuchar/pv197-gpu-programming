const int BLOCK_SIZE = 64;

#define P 8

__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    __shared__ volatile int shared[BLOCK_SIZE];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache[P];
    for (int j = 0; j < periods / P; j++) {
        for (int k = 0; k < P; k++) {
            cache[k] = changes[index + k * clients];
        }
        for (int k = 0; k < P; k++) {
            acc += cache[k];
            atomicAdd(&sum[j * P + k], acc);
            cache[k] = acc;
        }
        for (int k = 0; k < P; k++) {
            account[index + k * clients] = cache[k];
        }
        index += P * clients;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
