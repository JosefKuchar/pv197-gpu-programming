const int BLOCK_SIZE = 128;
__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    account[i] = changes[i];
    atomicAdd(&sum[0], account[i]);
    for (int j = 1; j < periods; j++) {
        account[j * clients + i] = account[(j - 1) * clients + i] + changes[j * clients + i];
        atomicAdd(&sum[j], account[j * clients + i]);
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
