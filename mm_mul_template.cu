#include <iostream>
#include <chrono>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

// Precision data types
typedef __half datatype_fl16;
typedef __nv_bfloat16 datatype_bf16;
typedef float datatype_fl32;
typedef double datatype_fl64;

template <typename T>
__global__ void matrixMultiply(T *A, T *B, T *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = static_cast<T>(0);

    if (row < N && col < N) {
        sum = 0.0;
        for (int k = 0; k < N; ++k) {
            T a = A[row * N + k];
            T b = B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

template <typename Datatype>
float runDatatype(int type) {
    
    const long N = 30000;
    size_t bytes = N * N * sizeof(Datatype);

    // Host matrices
    Datatype *h_A = new Datatype[N * N];
    Datatype *h_B = new Datatype[N * N];
    Datatype *h_C = new Datatype[N * N];

    // Initialize input matrices A and B
    for (int i = 0; i < N * N; ++i) {

        if (type == 1) {
            h_A[i] = (Datatype)((double)rand() / RAND_MAX); 
            h_B[i] = (Datatype)((double)rand() / RAND_MAX); 
        }
        else{
            h_A[i] = (Datatype)((float)rand() / RAND_MAX); 
            h_B[i] = (Datatype)((float)rand() / RAND_MAX); 
        }
    }


    // Device matrices
    Datatype *d_A, *d_B, *d_C;
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
    matrixMultiply<Datatype><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Start the timer
    // auto start = std::chrono::steady_clock::now();
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

    // Launch the kernel for actual computation
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Synchronize to ensure all kernels are finished
    

    // Stop the timer
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "Elapsed time: " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;

    // Allocate host memory for the result matrix
    Datatype *h_C_result = new Datatype[N * N];

    // Copy result matrix from device to host
    cudaMemcpy(h_C_result, d_C, N * N * sizeof(Datatype), cudaMemcpyDeviceToHost);

    // Print the result matrix
    // for(int i = 0; i < N; ++i)
    // {
    //     for(int j = 0; j < N; ++j)
    //     {
    //         printf("%.20f ", __float2half(h_C_result[i * N + j]));
            
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

    return milliseconds;
}

int main() {
    //run runDatatype with template datatype
    printf("------------------------------------\n");
    printf("Program: Matrix Matrix Mulitplication\n");
    float time = runDatatype<datatype_fl16>(0);
    printf("Elapsed time in milliseconds (FP16): %f \n", time);
    time = runDatatype<datatype_bf16>(0);
    printf("Elapsed time in milliseconds (BF16): %f \n", time);
    time = runDatatype<datatype_fl32>(0);
    printf("Elapsed time in milliseconds (FP32): %f \n", time);
    float time = runDatatype<datatype_fl64>(0);
    printf("Elapsed time in milliseconds (FP64): %f \n", time);
    return 0;
}