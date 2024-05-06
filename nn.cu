#include <iostream>
#include <random>
#include "cuda_fp16.h"
#include "cuda_bf16.h"


// Precision data types
typedef __half datatype_fl16;
typedef __nv_bfloat16 datatype_bf16;
typedef float datatype_fl32;
typedef double datatype_fl64;
cudaEvent_t start, stop;


// Kernel to perform 2D convolution using tensor cores
template <typename T>
__global__ void convolutionKernel(const T* __restrict__ input, const T* __restrict__ filter,
                                  T* __restrict__ output, int n_inputs, int n_filters,
                                  int height, int width, int filter_height, int filter_width) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int n = bx * blockDim.x + tx;
    const int m = by * blockDim.y + ty;

    T sum = static_cast<T>(0);
    if (n < n_filters && m < height * width) {
        const int output_x = m % width;
        const int output_y = m / width;

        for (int i = 0; i < n_inputs; ++i) {
            for (int j = 0; j < filter_height; ++j) {
                for (int k = 0; k < filter_width; ++k) {
                    const int input_x = output_x + k - (filter_width / 2);
                    const int input_y = output_y + j - (filter_height / 2);
                    if (input_x >= 0 && input_x < width && input_y >= 0 && input_y < height) {
                        const int input_index = i * height * width + input_y * width + input_x;
                        const int filter_index = i * filter_height * filter_width + j * filter_width + k;
                        sum += static_cast<T>(input[input_index] * filter[n * n_inputs * filter_height * filter_width + filter_index]);
                    }
                }
            }
        }
        output[m * n_filters + n] = sum;
    }
}

template <typename datatype>
float runDatatype(){
    const int n_inputs = 3;       // Number of input channels
    const int n_filters = 256;     // Number of filters
    const int height = 224;       // Input height
    const int width = 224;        // Input width
    const int filter_height = 3;  // Filter height
    const int filter_width = 3;   // Filter width

    // const int n_inputs = 6;       // Number of input channels
    // const int n_filters = 128;     // Number of filters
    // const int height = 448;       // Input height
    // const int width = 448;        // Input width
    // const int filter_height = 6;  // Filter height
    // const int filter_width = 6;   // Filter width


    

    const size_t input_size = n_inputs * height * width;
    const size_t filter_size = n_filters * n_inputs * filter_height * filter_width;
    const size_t output_size = n_filters * height * width;

    // Allocate host memory
    datatype* h_input = new datatype[input_size];
    datatype* h_filter = new datatype[filter_size];
    datatype* h_output = new datatype[output_size];

    // Initialize input and filter data
    // ...
    // NONE AT THE MOMENT... still runs fine ... ??

    // Allocate device memory
    datatype* d_input;
    datatype* d_filter;
    datatype* d_output;
    cudaMalloc(&d_input, input_size * sizeof(datatype));
    cudaMalloc(&d_filter, filter_size * sizeof(datatype));
    cudaMalloc(&d_output, output_size * sizeof(datatype));

    // Copy input and filter data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size * sizeof(datatype), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    const int tile_width = 16;
    const int tile_height = 16;
    const dim3 block_dim(tile_width, tile_height);
    const dim3 grid_dim((n_filters + tile_width - 1) / tile_width, (height * width + tile_height - 1) / tile_height);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel with different precision modes
    cudaEventRecord(start);
    convolutionKernel<datatype><<<grid_dim, block_dim>>>(d_input, d_filter, d_output, n_inputs, n_filters, height, width, filter_height, filter_width);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output data from device to host (omitted for brevity)
    // ...

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;

    return milliseconds;
}

int main() {
    //run runDatatype with template datatype
    printf("------------------------------------\n");
    printf("Program: Neural Network (no training data)\n");
    float time = runDatatype<datatype_fl16>();
    printf("Elapsed time in milliseconds (FP16): %f \n", time);
    time = runDatatype<datatype_bf16>();
    printf("Elapsed time in milliseconds (BF16): %f \n", time);
    time = runDatatype<datatype_fl32>();
    printf("Elapsed time in milliseconds (FP32): %f \n", time);
    time = runDatatype<datatype_fl64>();
    printf("Elapsed time in milliseconds (FP64): %f \n", time);
    return 0;
}