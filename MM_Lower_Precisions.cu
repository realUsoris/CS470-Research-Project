#include <iostream>
#include <chrono>
#include "cuda_fp16.h"
#include "cuda_bf16.h"


#ifdef USE_FP16
     typedef __half Datatype;
#elif USE_BF16
    typedef __nv_bfloat16 Datatype;
#elif USE_FP32
    typedef float Datatype;
#elif USE_FP64
    typedef double Datatype;
#endif

double frobeniusNorm(Datatype* matrix, int N) {
    double norm = 0.0;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) { 
            double val = ((double)matrix[i * N + j]); 
            norm += val * val;
        }
    }
    return sqrt(norm);
}

__global__ void matrixMultiply(Datatype *A, Datatype *B, Datatype *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        #ifdef USE_FP16
            Datatype sum = __float2half(0.0f);
        #elif USE_BF16
            Datatype sum = __float2bfloat16(0.0f);
        #elif USE_FP32
            Datatype sum = 0.0f;
        #elif USE_FP64
            Datatype sum = (double)0.0;
        #endif  
        
        for (int k = 0; k < N; ++k) {
            Datatype a = A[row * N + k];
            Datatype b = B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    
    if (argc != 2) {
        fprintf(stderr, "Usage: [N]\n");
        return 1;
    }

    const long N = strtol(argv[1], NULL, 10); 
    size_t bytes = N * N * sizeof(Datatype);
    

    // Host matrices
    Datatype *h_A = new Datatype[N * N];
    Datatype *h_B = new Datatype[N * N];
    Datatype *h_C = new Datatype[N * N];

    // Initialize input matrices A and B
    for (int i = 0; i < N * N; ++i) {

        #ifdef USE_FP16
            h_A[i] = __float2half((float)rand() / RAND_MAX); 
            h_B[i] = __float2half((float)rand() / RAND_MAX); 
        #elif USE_BF16
            h_A[i] = __float2bfloat16((float)rand() / RAND_MAX); 
            h_B[i] = __float2bfloat16((float)rand() / RAND_MAX); 
        #elif USE_FP32
            h_A[i] = ((float)rand() / RAND_MAX); 
            h_B[i] = ((float)rand() / RAND_MAX);
        #elif USE_FP64
            h_A[i] = ((double)rand() / RAND_MAX); 
            h_B[i] = ((double)rand() / RAND_MAX);
        #endif
    }

    #ifdef USE_FP16
        printf("Using FP16\n");
    #elif USE_BF16
        printf("Using BF16\n");
    #elif USE_FP32
        printf("Using FP32\n");
    #elif USE_FP64
        printf("Using FP64\n");
    #endif

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
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
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

    printf("[+] GPU Elapsed Time: %lf s Size: %d N: %d\n", milliseconds / 1000, sizeof(Datatype), N);

    // Synchronize to ensure all kernels are finished
    

    // Stop the timer
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "Elapsed time: " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;

    // Allocate host memory for the result matrix
    Datatype *h_C_result = new Datatype[N * N];

    // Copy result matrix from device to host
    cudaMemcpy(h_C_result, d_C, N * N * sizeof(Datatype), cudaMemcpyDeviceToHost);

    

    // Compute the Frobenius norm of the result matrix
    double result = frobeniusNorm(h_C_result, N);

    printf("Frobenius norm of the result matrix: %lf\n", result);

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

    return 0;
}
