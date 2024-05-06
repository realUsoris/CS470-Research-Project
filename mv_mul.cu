#include <iostream>
#include <chrono>
#include <random>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

// Precision data types
typedef __half datatype_fl16;
typedef __nv_bfloat16 datatype_bf16;
typedef float datatype_fl32;
typedef double datatype_fl64;
cudaEvent_t start, stop;

// Kernel to perform matrix-vector multiplication using tensor cores
template <typename T>
__global__ void matVecMultKernel(const T* __restrict__ matrix, const T* __restrict__ vector,
                                 T* __restrict__ result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        T sum = static_cast<T>(0);
        for (int col = 0; col < cols; ++col) {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

template <typename datatype>
double L2Norm(datatype* vector, int N) {
    double norm = 0.0;
    for(int i = 0; i < N; ++i) {
        double val = ((double)vector[i]); 
        norm += val * val;
    }
    return sqrt(norm);
}

template <typename datatype>
float runDatatype(long N){
    const int rows = N;  // Number of rows in the matrix
    const int cols = N;   // Number of columns in the matrix

    // Allocate host memory
    datatype* h_matrix = new datatype[rows * cols];
    datatype* h_vector = new datatype[cols];
    datatype* h_result = new datatype[rows];

    // Initialize matrix and vector data with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < rows * cols; ++i) {
        h_matrix[i] = dis(gen);
    }
    for (int i = 0; i < cols; ++i) {
        h_vector[i] = dis(gen);
    }

    // Allocate device memory
    datatype* d_matrix;
    datatype* d_vector;
    datatype* d_result;
    cudaMalloc(&d_matrix, rows * cols * sizeof(datatype));
    cudaMalloc(&d_vector, cols * sizeof(datatype));
    cudaMalloc(&d_result, rows * sizeof(datatype));

    // Copy input data to device
    cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, cols * sizeof(datatype), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    const int blockSize = 256;
    const int gridSize = (rows + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel with different precision modes
    cudaEventRecord(start);
    matVecMultKernel<datatype><<<gridSize, blockSize>>>(d_matrix, d_vector, d_result, rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Allocate host memory for the result matrix
    datatype *h_C_result = new datatype[N];

    // Copy result matrix from device to host
    cudaMemcpy(h_C_result, d_result, N * sizeof(datatype), cudaMemcpyDeviceToHost);

    // Compute the Frobenius norm of the result matrix
    double result = L2Norm(h_C_result, N);
    printf("L2 norm of the result vector: %lf\n", result);


    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    // Free host memory
    delete[] h_matrix;
    delete[] h_vector;
    delete[] h_result;

    return milliseconds;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        fprintf(stderr, "Usage: [N]\n");
        return 1;
    }

    const long N = strtol(argv[1], NULL, 10); 

    //run runDatatype with template datatype_hf
    printf("------------------------------------\n");
    printf("Program: Matrix Vector Mulitplication\n");
    printf("Matrix Size: %ld x %ld\n", N, N);
    float time = runDatatype<datatype_fl16>(N);
    printf("Elapsed time in milliseconds (FP16): %f \n", time);
    time = runDatatype<datatype_bf16>(N);
    printf("Elapsed time in milliseconds (BF16): %f \n", time);
    time = runDatatype<datatype_fl32>(N);
    printf("Elapsed time in milliseconds (FP32): %f \n", time);
    time = runDatatype<datatype_fl64>(N);
    printf("Elapsed time in milliseconds (FP64): %f \n", time);
    return 0;
}