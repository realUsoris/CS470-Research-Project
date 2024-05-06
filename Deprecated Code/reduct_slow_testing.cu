#include <iostream>
#include <chrono>
#include <random>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

// Define precision data types

typedef __half datatype;
// typedef __nv_bfloat16 datatype;
// typedef float datatype;
// typedef double datatype;

// Kernel to perform parallel reduction using tensor cores
template <typename T>
__global__ void parallelReductionKernel(const T* __restrict__ input, T* __restrict__ output, int size) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : static_cast<T>(0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int size = 100000000;  // Size of the input array

    // Allocate host memory
    datatype* h_input = new datatype[size];
    datatype* h_output = new datatype[size / 256];

    // Initialize input data with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

    for (int i = 0; i < size; ++i) {
        h_input[i] = dis(gen);
    }

    // Allocate device memory
    datatype* d_input;
    datatype* d_output;
    cudaMalloc(&d_input, size * sizeof(datatype));
    cudaMalloc(&d_output, (size / 256) * sizeof(datatype));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(datatype), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;

    // Launch kernel with different precision modes
    auto start = std::chrono::steady_clock::now();
    parallelReductionKernel<datatype><<<gridSize, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time (FP16): " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;
    // std::cout << "Elapsed time (BF16): " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;
    // std::cout << "Elapsed time (FP32): " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;
    // std::cout << "Elapsed time (FP64): " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;

    // Copy output data from device to host (omitted for brevity)
    // ...

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    return 0;
}