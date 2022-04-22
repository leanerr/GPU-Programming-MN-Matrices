/* File:     mat_add.cu
 * Purpose:  Implement matrix addition on a gpu using cuda
 *
 * Compile:  nvcc [-g] [-G] -arch=sm_21 -o mat_add mat_add.cu 
 * Run:      ./mat_add <10> <10>
 *              m is the number of rows
 *              n is the number of columns
 *
 * Input:    The matrices A and B
 * Output:   Result of matrix addition.  
 *
 * Notes:
 * 1.  CUDA is installed on all of the machines in HR 530, HR 235, and
 *     and LS G12
 * 2.  If you get something like "nvcc: command not found" when you try
 *     to compile your program.  Type the following command
 *
 *           $ export PATH=/usr/local/cuda/bin:$PATH
 *
 *     (As usual the "$" is the shell prompt:  just type the rest 
 *     of the line.)
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 
 /*---------------------------------------------------------------------
  * Kernel:   Mat_add
  * Purpose:  Implement matrix addition
  * In args:  A, B, m, n
  * Out arg:  C
  */
 __global__ void Mat_add(float A[], float B[], float C[], int m, int n) {
    /* blockDim.x = threads_per_block                            */
    /* First block gets first threads_per_block components.      */
    /* Second block gets next threads_per_block components, etc. */
    int my_ij = blockDim.x * blockIdx.x + threadIdx.x;
 
    /* The test shouldn't be necessary */
    if (blockIdx.x < m && threadIdx.x < n) 
       C[my_ij] = A[my_ij] + B[my_ij];
 }  /* Mat_add */
 
 
 /*---------------------------------------------------------------------
  * Function:  Read_matrix
  * Purpose:   Read an m x n matrix from stdin
  * In args:   m, n
  * Out arg:   A
  */
 void Read_matrix(float A[], int m, int n) {
    int i, j;
 
    for (i = 0; i < m; i++)
       for (j = 0; j < n; j++)
          scanf("%f", &A[i*n+j]);
 }  /* Read_matrix */
 
 
 /*---------------------------------------------------------------------
  * Function:  Print_matrix
  * Purpose:   Print an m x n matrix to stdout
  * In args:   title, A, m, n
  */
 void Print_matrix(char title[], float A[], int m, int n) {
    int i, j;
 
    printf("%s\n", title);
    for (i = 0; i < m; i++) {
       for (j = 0; j < n; j++)
          printf("%.1f ", A[(i*n)+j]);
       printf("\n");
    }  
 }  /* Print_matrix */
 
 
 /* Host code */
 int main(int argc, char* argv[]) {
    int m, n;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size;
 
    /* Get size of matrices */
    if (argc != 3) {
       fprintf(stderr, "usage: %s <row count> <col count>\n", argv[0]);
       exit(0);
    }
    m = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);
    printf("m = %d, n = %d\n", m, n);
    size = m*n*sizeof(float);
 
    h_A = (float*) malloc(size);
    h_B = (float*) malloc(size);
    h_C = (float*) malloc(size);
    
    printf("Enter the matrices A and B\n");
    Read_matrix(h_A, m, n);
    Read_matrix(h_B, m, n);
 
    Print_matrix("A =", h_A, m, n);
    Print_matrix("B =", h_B, m, n);
 
    /* Allocate matrices in device memory */
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
 
    /* Copy matrices from host memory to device memory */
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
 
    /* Invoke kernel using m thread blocks, each of    */
    /* which contains n threads                        */
    Mat_add<<<m, n>>>(d_A, d_B, d_C, m, n);
 
    /* Wait for the kernel to complete */
    cudaThreadSynchronize();
 
    /* Copy result from device memory to host memory */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
 
    Print_matrix("The sum is: ", h_C, m, n);
 
    /* Free device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
 
    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);
 
    return 0;
 }  /* main */