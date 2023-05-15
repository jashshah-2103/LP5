//From non cuda machine
// ssh exam@10.10.12.68
// password : xxxxx
// cat >> file.cu
// paste code

// ^D

// nvcc file.cu
// ./a.out

%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20) // Vector size
#define THREADS_PER_BLOCK 512

// Parallel vector addition kernel
__global__ void add_kernel(float* a, float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

int main()
{
    float *a, *b, *c; // Host vectors
    float *d_a, *d_b, *d_c; // Device vectors
    int size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize host vectors with random data
    for (int i = 0; i < N; i++)
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

    // Perform parallel vector addition
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    add_kernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Parallel addition time: %f ms\n", time);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool flag=true;
    for (int i = 0; i < N; i++)
    {
        if (c[i] != a[i] + b[i])
        {
            printf("Error: incorrect result at index %d\n", i);
            flag=false;
            break;
        }
    }
    if(flag){
        printf("Success : correct result\n");
    }

    // Perform serial vector addition
    cudaEventRecord(start, 0);
    
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
        
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Serial addition time: %f ms\n", time);

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
 
// // CUDA kernel. Each thread takes care of one element of c
// __global__ void vecAdd(double *a, double *b, double *c, int n)
// {
//     // Get our global thread ID
//     int id = blockIdx.x*blockDim.x+threadIdx.x;
 
//     // Make sure we do not go out of bounds
//     if (id < n)
//         c[id] = a[id] + b[id];
// }
 
// int main( int argc, char* argv[] )
// {
//     // Size of vectors
//     int n = 100000;
 
//     // Host input vectors
//     double *h_a;
//     double *h_b;
//     //Host output vector
//     double *h_c;
 
//     // Device input vectors
//     double *d_a;
//     double *d_b;
//     //Device output vector
//     double *d_c;
 
//     // Size, in bytes, of each vector
//     size_t bytes = n*sizeof(double);
 
//     // Allocate memory for each vector on host
//     h_a = (double*)malloc(bytes);
//     h_b = (double*)malloc(bytes);
//     h_c = (double*)malloc(bytes);
 
//     // Allocate memory for each vector on GPU
//     cudaMalloc(&d_a, bytes);
//     cudaMalloc(&d_b, bytes);
//     cudaMalloc(&d_c, bytes);
 
//     int i;
//     // Initialize vectors on host
//     for( i = 0; i < n; i++ ) {
//         h_a[i] = sin(i)*sin(i);
//         h_b[i] = cos(i)*cos(i);
//     }
 
//     // Copy host vectors to device
//     cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
//     int blockSize, gridSize;
 
//     // Number of threads in each thread block
//     blockSize = 1024;
 
//     // Number of thread blocks in grid
//     gridSize = (int)ceil((float)n/blockSize);
 
//     // Execute the kernel
//     vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
//     // Copy array back to host
//     cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
//     // Sum up vector c and print result divided by n, this should equal 1 within error
//     double sum = 0;
//     for(i=0; i<n; i++)
//         sum += h_c[i];
//     printf("final result: %f\n", sum/n);
 
//     // Release device memory
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
 
//     // Release host memory
//     free(h_a);
//     free(h_b);
//     free(h_c);
 
//     return 0;
// }
// #include <iostream>
// #include <cstdlib>
// #include <ctime>
// #include <omp.h>

// using namespace std;

// const int VECTOR_SIZE = 100;
// int main()
// {
//     // initialize random seed
//     srand(time(NULL));
//     // allocate memory for the vectors 
//     int* vector1 = new int[VECTOR_SIZE]; 
//     int* vector2 = new int[VECTOR_SIZE]; 
//     int* result = new int[VECTOR_SIZE];
//     // fill the vectors with random numbers 
//     #pragma omp parallel for
//     for (int i = 0; i < VECTOR_SIZE; i++)
//     {
//         vector1[i] = rand() % 10000;
//         vector2[i] = rand() % 10000;
//     }
//     // add the vectors in parallel using OpenMP 
//     #pragma omp parallel for
//     for (int i = 0; i < VECTOR_SIZE; i++)
//     {
//         result[i] = vector1[i] + vector2[i];
//     }
//     // print the first and second vectors and their sum 
//     cout << "Vector 1: \n[";
//     for (int i = 0; i < VECTOR_SIZE; i++)
//     {
//         cout << vector1[i];
//         if (i != VECTOR_SIZE - 1)
//         {
//             cout << ", ";
//         }
//     }
//     cout << "]" << endl;
//     cout << "Vector 2:\n [";
//     for (int i = 0; i < VECTOR_SIZE; i++)
//     {
//         cout << vector2[i];
//         if (i != VECTOR_SIZE - 1)
//         {
//             cout << ", ";
//         }
//     }
//     cout << "]" << endl;
//     cout << "Result: \n[";
//     for (int i = 0; i < VECTOR_SIZE; i++)
//     {
//         cout << result[i];
//         if (i != VECTOR_SIZE - 1)
//         {
//             cout << ", ";
//         }
//     }
//     cout << "]" << endl;
//     // free the allocated memory
//     delete[] vector1;
//     delete[] vector2;
//     delete[] result;
//     return 0;
// }
