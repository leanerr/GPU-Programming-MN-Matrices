/* File:     matrix_sum.cu
 * Purpose:  Implement matrix addition on a gpu using cuda
 *
 * Compile:  nvcc -o matrix_sum matrix_sum.cu 
 * Run:      ./matrix_sum <10> <10>
 *              m = the number of rows
 *              n = the number of columns
 *
 * Input:    The matrixes A and B
 * Output:   Result of matrix addition.  
 *
 */
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 
 /*---------------------------------------------------------------------
  * Kernel:   matrix_sum
  * Purpose:  Implement matrix addition
  * In args:  A, B, m, n
  * Out arg:  C
  */
 __global__ void matrix_sum(float A[], float B[], float C[], int m, int n) {
    /* blockDim.x = threads_per_block                            */
    /* First block gets first threads_per_block components.      */
    /* Second block gets next threads_per_block components, etc. */
    int my_ij = blockDim.x * blockIdx.x + threadIdx.x; 
 
    /* The test shouldn't be necessary */
    if (blockIdx.x < m && threadIdx.x < n) 
       C[my_ij] = A[my_ij] + B[my_ij];
 }  /* matrix_sum */
 
 
 /*---------------------------------------------------------------------
  get_matrix
  */
  void get_matrix(float A[], int m, int n) {
    int i, j;
 
    for (i = 0; i < m; i++)
       for (j = 0; j < n; j++)
          scanf("%f", &A[i*n+j]);
 }  /* get_matrix */
 
 
 /*---------------------------------------------------------------------
	Print_matrix
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
 
 
 /* Host code - CPU  */
 int main(int argc, char* argv[]) {
    int m, n;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size;
 
    /* Get size of matrixes */
    if (argc != 3) {
       fprintf(stderr, "usage: %s <row count> <col count>\n", argv[0]);
       exit(0);
    }
    m = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);
    printf("m = %d, n = %d\n", m, n);
    size = m*n*sizeof(float);

   /* declare pointers to vectors in device memory and allocate memory */
    h_A = (float*) malloc(size); 
    h_B = (float*) malloc(size);
    h_C = (float*) malloc(size);
    
    printf("Enter some numbers and we create 2 matrix , A and B , first A : \n");
    get_matrix(h_A, m, n);
    printf("Enter second matrix: ");
    get_matrix(h_B, m, n);
 
    Print_matrix("A =", h_A, m, n);
    Print_matrix("B =", h_B, m, n);
 
    /* Allocate matrixes in device memory */
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
 
    /* Copy matrixes from host memory to device memory */
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
 
    /* Invoke kernel using m thread blocks, each of    */
    /* which contains n threads                        */
    dim3 block_size( 16, 16 );
    dim3 num_blocks( ( n - 1 + block_size.x ) / block_size.x, 
                     ( m - 1 + block_size.y ) / block_size.y );
  
    matrix_sum<<<block_size, num_blocks>>>(d_A, d_B, d_C, m, n);
 
    /* Wait for the kernel to complete */
    cudaThreadSynchronize();
 
    /* Copy result from device memory to host memory */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
 
    Print_matrix("The result is: ", h_C, m, n);
 
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