#include<stdio.h>
#include<cuda_runtime.h>
#include <time.h>

#define M 512
#define N 512
#define K 512
#define THREAD_PER_BLOCK 32


__global__ void gemm(int* a, int* b, int* c) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //blockIdx.y * blockDim.y means I move to the leftmost of 
    //the block in which the element resides 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //similar to row

    // Check boundaries to ensure within matrix dimensions
    if (row < M && col < N) {
        int result = 0;
        // Perform matrix multiplication for the given element (row, col)
        for (int k = 0; k < K; ++k) {
            result += a[row * K + k] * b[k * N + col];
            //Notice that I use 1-dimensional array to 
            //simulate 2-dimensional array
        }
        // Store the result in matrix C, also use 1-dimensional array to 
        //simulate 2-dimensional array
        c[row * N + col] = result;
    }
}

int main(){
    //Host memory allocation
    int *a, *b, *c, *c_cmp;
    a = (int*) malloc(M * K * sizeof(int));
    b = (int*) malloc(K * N * sizeof(int));
    c = (int*) malloc(M * N * sizeof(int));
    c_cmp = (int*) malloc(M * N * sizeof(int));
    //matrix initialization
    srand((unsigned)time(NULL)); 
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            a[i * K + j] = rand() % 100;
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            b[i * N + j] = rand() % 100;
        }
        
    }
    //Device memory allocation
    int *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, M * K * sizeof(int));
    cudaMalloc(&b_d, K * N * sizeof(int));
    cudaMalloc(&c_d, M * N * sizeof(int));
    cudaMemcpy(a_d, a, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
    //create a "dim3" object named block, which is a 2D block with dimension
    //(THREAD_PER_BLOCK, THREAD_PER_BLOCK)
    dim3 grid(M/THREAD_PER_BLOCK, N/THREAD_PER_BLOCK);
    //This line create a "dim3" object named grid which is also 
    //a 2D grid with dimension (M/THREAD_PER_BLOCK, N/THREAD_PER_BLOCK)
    gemm<<<grid, block>>>(a_d, b_d, c_d);
    //launch the "gemm" kernel with block and grid configuration
    //and pass a_d, b_d, c_d as parameters

    cudaMemcpy(c, c_d, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    
    //compute the result using CPU
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < K; k++){
                c_cmp[i * N + j] += a[i * K + k] * b[k * N + j]; 
            }
        }
    }
    bool flag = 1;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(c_cmp[i * N + j] != c[i * N + j]){
                flag = 0;
                break;
            }
        }
    }
    if(flag){
        printf("result correct\n");
    }
    else{
        printf("result wrong\n");
    }
    free(a);
    free(b);
    free(c);
    free(c_cmp);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}