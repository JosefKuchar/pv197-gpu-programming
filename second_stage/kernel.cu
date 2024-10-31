const int BLOCK_SIZE = 32;
const int D = 32;

__global__ void kernel(int* changes, int* account, int* sum, int clients, int periods) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = threadIdx.y * periods / D;
    account[i + offset * clients] = changes[i + offset * clients];
    for (int j = offset + 1; j < offset + periods / D; j++) {
        account[j * clients + i] = account[(j - 1) * clients + i] + changes[j * clients + i];
    }
}

void solveGPU(int* changes, int* account, int* sum, int clients, int periods) {
    dim3 block(BLOCK_SIZE, D);
    kernel<<<clients / BLOCK_SIZE, block>>>(changes, account, sum, clients, periods);
    // kernel2<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
