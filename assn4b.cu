// !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
// %load_ext nvcc_plugin

%%cu
#include <stdio.h>
#include <stdlib.h>
#define N 1024 
#define BLOCK_SIZE 32
__global__ void matrixMultiply(float *a, float *b, float *c, int n) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < n; k++) {
      sum += a[row * n + k] * b[k * n + col]; 
      }
    c[row * n + col] = sum; 
}
int main() {
    
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;

  int size = N * N * sizeof(float);

  // Allocate memory on host 
  
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  // Initialize matrices
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) { 
        a[i * N + j] = i + j;
        b[i * N + j] = i - j;
    } 
  }
  // Allocate memory on device 
  cudaMalloc((void **)&d_a, size); 
  cudaMalloc((void **)&d_b, size); 
  cudaMalloc((void **)&d_c, size);

  // Copy matrices from host to device 
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Define grid and block sizes
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);

  // Launch kernel on device
  matrixMultiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

  // Copy result from device to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < N; k++) {
        sum += a[i * N + k] * b[k * N + j]; 
        }
    if (c[i * N + j] != sum) {
      printf("Error: c[%d][%d] = %f\n", i, j, c[i * N + j]); break;
     } 
    }
  }
  // Free memory on host and device 
  
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

//##############################################################################################

%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 10) // Matrix size
#define THREADS_PER_BLOCK 32

// Parallel matrix multiplication kernel
__global__ void multiply_kernel(float* a, float* b, float* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sum += a[row * N + i] * b[i * N + col];
        }

        c[row * N + col] = sum;
    }
}


int main()
{
    float *a, *b, *c; // Host matrices
    float *d_a, *d_b, *d_c; // Device matrices
    int size = N * N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize host matrices with random data
    for (int i = 0; i < N * N; i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host data to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Perform parallel matrix multiplication
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks(N / THREADS_PER_BLOCK, N / THREADS_PER_BLOCK);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Parallel multiplication time: %f ms\n", time);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            if (c[i * N + j] != sum)
            {
                printf("Error: incorrect result at index (%d, %d)\n", i, j);
                break;
            }
        }
    }
 
 
    // Perform serial matrix multiplication
    cudaEventRecord(start, 0);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("\nSerial addition time: %f ms\n", time);
 
    // Display first 10x10 elements of result
    printf("Result:\n");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%.2f ", c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
