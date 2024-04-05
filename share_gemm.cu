#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
//#include <iostream>

#define M 1024
#define N 1024
#define K 1024

#define THREAD_PER_BLOCK 32

//define a struct to represent the matrix
//This is for convenience of banking
typedef struct
{
    int Num_of_Rows;
    int Num_of_Cols;
    int stride;
    int * Mat;
} Matrix;

__global__ void gemm(Matrix a, Matrix b, Matrix c) {
    //compute the bank index
    int B_Row = blockIdx.y;
    int B_Col = blockIdx.x;

    //define the atarget this thread is currently on
    Matrix Sub_of_C;
    Sub_of_C.Num_of_Rows = THREAD_PER_BLOCK;
    Sub_of_C.Num_of_Cols = THREAD_PER_BLOCK;
    Sub_of_C.stride = c.stride;
    Sub_of_C.Mat = &(c.Mat[B_Row * THREAD_PER_BLOCK * c.stride + THREAD_PER_BLOCK * B_Col]);

    //result is a temporary variable which represent the result for the element
    //at (row, col)
    int result = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    //Use a for loop to do the computing
    //looping over the blocks
    for(int k = 0; k < N / THREAD_PER_BLOCK; k++)
    {
        //define the sub matrix of A and B
        Matrix Sub_of_A, Sub_of_B;
        Sub_of_A.Num_of_Rows = THREAD_PER_BLOCK;
        Sub_of_A.Num_of_Cols = THREAD_PER_BLOCK;
        Sub_of_A.stride = a.stride;
        Sub_of_A.Mat = &(a.Mat[a.stride * B_Row * THREAD_PER_BLOCK + k * THREAD_PER_BLOCK]);
    
        Sub_of_B.Num_of_Rows = THREAD_PER_BLOCK;
        Sub_of_B.Num_of_Cols = THREAD_PER_BLOCK;
        Sub_of_B.stride = b.stride;
        Sub_of_B.Mat = &(b.Mat[b.stride * k * THREAD_PER_BLOCK + B_Col * THREAD_PER_BLOCK]);

        //Pushing elements into the shared memory
        __shared__ int Shared_sub_matrix_A[THREAD_PER_BLOCK][THREAD_PER_BLOCK];
        __shared__ int Shared_sub_matrix_B[THREAD_PER_BLOCK][THREAD_PER_BLOCK];

        //In every cycle, each thread is responsible for pushing one element
        Shared_sub_matrix_A[row][col] = Sub_of_A.Mat[row * a.stride + col];
        Shared_sub_matrix_B[row][col] = Sub_of_B.Mat[row * b.stride + col];
        //Syncranize, waiting for them to complete filling the shared memory
        __syncthreads();

        //Compute the product and add them to the temporary variable
        for(int r = 0; r < THREAD_PER_BLOCK; r++)
        {
            result += Shared_sub_matrix_A[row][r] * Shared_sub_matrix_B[r][col];
        }

        //syncranize, waiting for them to complete computing
        __syncthreads();

    }    
    //Now assign the temporary element to Matrix C
    Sub_of_C.Mat[row * Sub_of_C.stride + col] = result;
}

int main() {
    // Host memory allocation
    int *a, *b, *c, *c_cmp;
    a = (int*)malloc(M * K * sizeof(int));
    b = (int*)malloc(K * N * sizeof(int));
    c = (int*)malloc(M * N * sizeof(int));
    c_cmp = (int*)malloc(M * N * sizeof(int));

    // Matrix initialization
    srand((unsigned)time(NULL));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            a[i * K + j] = rand() % 100;
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i * N + j] = rand() % 100;
        }
    }

    // Device memory allocation
    Matrix a_d, b_d, c_d; 

    a_d.Num_of_Rows = M;
    a_d.stride = M;
    a_d.Num_of_Cols = K;

    b_d.Num_of_Rows = K;
    b_d.stride = K;
    b_d.Num_of_Cols = N;

    c_d.Num_of_Rows = M;
    c_d.stride = M; 
    c_d.Num_of_Cols = N;

    cudaMalloc(&a_d.Mat, M * K * sizeof(int));
    cudaMalloc(&b_d.Mat, K * N * sizeof(int));
    cudaMalloc(&c_d.Mat, M * N * sizeof(int));
    cudaMemcpy(a_d.Mat, a, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d.Mat, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    dim3 grid(N / block.x, M / block.y);

    // Launch the CUDA kernel with shared memory
    gemm<<<grid, block>>>(a_d, b_d, c_d);

    // Copy results back to host
    cudaMemcpy(c, c_d.Mat, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    
   // Compute the result using CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                c_cmp[i * N + j] += a[i * K + k] * b[k * N + j];
            }
        }
    }

    // Check if the results match
    bool flag = true;
    for (int i = 0; i < M * N; ++i) {
        //std::cout << "i= " << i << "    GPU: " << c[i] << "  CPU:  " << c_cmp[i] << std::endl; 
        if (c[i] != c_cmp[i]) {
            flag = false;
            break;
        }
    }

    // Print the result
    if (flag) {
        printf("Result is correct\n");
    } else {
        printf("Result is incorrect\n");
    }

    // Free allocated memory
    free(a);
    free(b);
    free(c);
    free(c_cmp);
    cudaFree(a_d.Mat);
    cudaFree(b_d.Mat);
    cudaFree(c_d.Mat);

    return 0;
}
