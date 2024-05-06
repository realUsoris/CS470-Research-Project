#include <iostream>
#include <chrono>
#include "cuda_fp16.h"

// depreciated code

// Define half-precision floating-point data type
typedef __half float16;

// Kernel to perform matrix multiplication using tensor cores with half precision
__global__ void matrixMultiply(float16 *A, float16 *B, float16 *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float16 sum = __float2half(0.0f);
        for (int k = 0; k < N; ++k) {
            float16 a = A[row * N + k];
            float16 b = B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Matrix size
    int N = 10240; // Adjust matrix size according to your needs
    size_t bytes = N * N * sizeof(float16);

    // Host matrices
    float16 *h_A = new float16[N * N];
    float16 *h_B = new float16[N * N];
    float16 *h_C = new float16[N * N];

    // Initialize input matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = __float2half((float)rand() / RAND_MAX); // Initialize with some value
        h_B[i] = __float2half((float)rand() / RAND_MAX); // Initialize with some value
    }

    // Device matrices
    float16 *d_A, *d_B, *d_C;
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

    printf("Matrix multiplication completed\n");
    // Allocate host memory for the result matrix
    float16 *h_C_result = new float16[N * N];

    // Copy result matrix from device to host
    cudaMemcpy(h_C_result, d_C, N * N * sizeof(float16), cudaMemcpyDeviceToHost);

    // // Print the result matrix
    // for(int i = 0; i < N; ++i)
    // {
    //     for(int j = 0; j < N; ++j)
    //     {
    //         printf("%lf ", __half2float(h_C_result[i * N + j]));
    //     }
    //     printf("\n");
    // }

    // Free the host memory for the result matrix
    delete[] h_C_result;

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
