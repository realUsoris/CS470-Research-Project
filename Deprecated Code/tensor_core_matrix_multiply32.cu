#include <iostream>
#include <chrono>

// depreciated code

// Kernel to perform matrix multiplication using tensor cores
__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void printMatrix(float **matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f\t", matrix[i][j]); // Adjust the precision specifier as needed
        }
        printf("\n");
    }
}

int main() {
    // Matrix size
    int N = 10240; // Adjust matrix size according to your needs
    size_t bytes = N * N * sizeof(float);

    // Host matrices
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize input matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX; // Initialize with some value
        h_B[i] = (float)rand() / RAND_MAX; // Initialize with some value
    }

    // printMatrix(&h_A, N, N);
    // printf("\n\n\n");
    // printMatrix(&h_B, N, N);
    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    // Perform warm-up to ensure data is in GPU memory
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Start the timer
    auto start = std::chrono::steady_clock::now();

    // Launch the kernel for actual computation
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure all kernels are finished
    cudaDeviceSynchronize();

    // Stop the timer
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    //     // Allocate host memory for the result matrix
    // float *h_C_result = new float[N * N];

    // // Copy result matrix from device to host
    // cudaMemcpy(h_C_result, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // // Print the result matrix
    // for(int i = 0; i < N; ++i)
    // {
    //     for(int j = 0; j < N; ++j)
    //     {
    //         printf("%f ", h_C_result[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // // Free the host memory for the result matrix
    // delete[] h_C_result;

    // printMatrix(&h_C, N, N);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
