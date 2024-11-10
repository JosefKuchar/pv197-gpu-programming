const int BLOCK_SIZE = 32;

#define P 4

__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;
    int cache_in[P];
    int cache_out[P];
    for (int j = 0; j < periods / P; j++) {
        for (int k = 0; k < P; k++) {
            cache_in[k] = changes[index + k * clients];
        }
        for (int k = 0; k < P; k++) {
            acc += cache_in[k];
            // atomicAdd(&sum[j * P + k], acc);
            cache_out[k] = acc;
        }
        for (int k = 0; k < P; k++) {
            account[index + k * clients] = cache_out[k];
        }
        index += P * clients;
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
