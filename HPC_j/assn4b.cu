//From non cuda machine
// ssh exam@10.10.12.68
// password : xxxxx
// cat >> file.cu
// paste code

// ^D

// nvcc file.cu
// ./a.out

// !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
// %load_ext nvcc_plugin

#include<iostream>
#include<bits/stdc++.h>
#include<cuda.h>
#define BLOCK_SIZE 16

using namespace std;



void initialize_matrix(int *array, int rows, int cols){
    for(int i = 0 ; i < rows; i++){
        for(int j = 0; j < cols; j++){
            array[i*cols + j] = rand() % 10;
        }
    }
}

void print_matrix(int *array, int rows, int cols){
    for(int i = 0 ; i < rows; i++){
        for(int j = 0; j < cols; j++){
            cout << array[i*cols + j] << " ";
        }
        cout << endl;
    }
}

void matrix_multiplication_cpu(int *a, int *b, int *c, int common, int c_rows,int c_cols){
    for(int i = 0; i < c_rows; i++){
        for(int j = 0; j < c_cols; j++){
            int sum = 0;
            for(int k = 0; k < common; k++){
                sum += a[i*common + k] * b[k*c_cols + j];
            }
            c[i*c_cols + j] = sum;
        }
    }
}



__global__ void matrix_multiply(int *a, int *b, int *c, int c_rows, int common, int c_cols)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int sum=0;
   
    if(col < c_cols && row < c_rows) {
      for(int j = 0 ;j < common;j++)
      {
          sum += a[row*common+j] * b[j*c_cols+col];
      }
      c[c_cols*row+col]=sum;
    }
    
}


int main(){

    int A_rows, A_cols, B_rows, B_cols, C_rows, C_cols;
    cout << "Dimensions of matrix 1:\n";
    cout << "Rows: ";
    cin >> A_rows;
    cout << "Columns: ";
    cin >> A_cols;
    cout << "Dimensions of matrix 2:\n";
    cout << "Rows: " << A_cols << endl << "Columns: ";
    cin >> B_cols;
    B_rows = A_cols;
    C_rows = A_rows;
    C_cols = B_cols;

    int A_size = A_rows * A_cols;
    int B_size = B_rows * B_cols;
    int C_size = C_rows * C_cols;

    int *A, *B, *C;
    int *m1,*m2,*result;

    A = new int[A_size];
    B = new int[B_size];
    C = new int[C_size];

    initialize_matrix(A,A_rows,A_cols);
    cout << "Matrix 1\n";
    // print_matrix(A,A_rows,A_cols);
    initialize_matrix(B,B_rows,B_cols);
    cout << "Matrix 2\n";
    // print_matrix(B,B_rows,B_cols);

    cudaMallocManaged(&m1, A_size * sizeof(int));
    cudaMallocManaged(&m2, B_size * sizeof(int));
    cudaMallocManaged(&result, C_size * sizeof(int));

    cudaMemcpy(m1,A,A_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m2,B,B_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(A_rows + BLOCK_SIZE  - 1 / BLOCK_SIZE, B_cols + BLOCK_SIZE - 1 / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);

    float gpu_elapsed_time;
    cudaEvent_t gpu_start,gpu_stop;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);
    matrix_multiply<<<dimGrid,dimBlock>>>(m1,m2,result,C_rows,A_cols,C_cols);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cudaMemcpy(C,result,C_size*sizeof(int),cudaMemcpyDeviceToHost);
    cout << "GPU result:\n";
    // print_matrix(C,C_rows,C_cols);
    cout<<"GPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds"<<endl;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);
    matrix_multiplication_cpu(A,B,C,A_cols,C_rows,C_cols);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cout << "CPU result:\n";
    // print_matrix(C,C_rows,C_cols);
    cout<<"CPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds"<<endl;

    cudaFree(m1);
    cudaFree(m2);
    cudaFree(result);

    return 0;
}


//____________________________________________________________________________________________________



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
