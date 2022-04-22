// $Smake: nvcc -arch=sm_30 -O2 -o %F %f
//
// add-vectors.cu - addition of two arrays on GPU device
//
// This program follows a very standard pattern:
//  1) allocate memory on host
//  2) allocate memory on device
//  3) initialize memory on host
//  4) copy memory from host to device
//  5) execute kernel(s) on device
//  6) copy result(s) from device to host
//
// Note: it may be possible to initialize memory directly on the device,
// in which case steps 3 and 4 are not necessary, and step 1 is only
// necessary to allocate memory to hold results.

#include <stdio.h>
#include <cuda.h>

//-----------------------------------------------------------------------------
// Kernel that executes on CUDA device

__global__ void add_vectors(
    float **c,      // out - pointer to result vector c
    float **a,      // in  - pointer to summand vector a
    float **b,      // in  - pointer to summand vector b
    int n ,      // in  - vector length
    int m
    )
{
    // Assume single block grid and 1-D block
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only do calculation if we have real data to work with
    if ( idx < n ) c[idx] = a[idx] + b[idx];
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Main program executes on host device

int main( int argc, char* argv[] )
{
    // determine vector length
    int n = 10;  
    int m = 10;   // set default length
    if ( argc > 1 )
    {
        n = atoi( argv[1] );  // override default length
        if ( n <= 0 )
        {
            fprintf( stderr, "Vector length must be positive\n" );
            return EXIT_FAILURE;
        }
        
    }

    // determine vector size in bytes
    const size_t vector_size = n * m sizeof( float );
 //   const size_t matrix_col_size = n * sizeof( float );
  //  const size_t matrix_row_size = m * sizeof( float );



    // declare pointers to vectors in host memory and allocate memory
    float *a, *b, *c;
    a = (float*) malloc( vector_size );
    b = (float*) malloc( vector_size );
    c = (float*) malloc( vector_size );

    // declare pointers to vectors in device memory and allocate memory
    float *a_d, *b_d, *c_d;
    cudaMalloc( (void**) &a_d, vector_size );
    cudaMalloc( (void**) &b_d, vector_size );
    cudaMalloc( (void**) &c_d, vector_size );

    // initialize vectors and copy them to device
    for ( int i = 0; i < n; i++ )
    {
        a[i] =   1.0 * i;
        b[i] = 100.0 * i;
    }
    cudaMemcpy( a_d, a, vector_size, cudaMemcpyHostToDevice );
    cudaMemcpy( b_d, b, vector_size, cudaMemcpyHostToDevice );

    // do calculation on device
   // int block_size = (16,16);
    dim3 block_size( 16, 16 );
    dim3 num_blocks( ( n - 1 + block_size.x ) / block_size.x, 
                    ( m - 1 + block_size.y ) / block_size.y );
    add_matrices<<< num_blocks, block_size >>>( c_d, a_d, b_d, m, n );


    // retrieve result from device and store on host
    cudaMemcpy( c, c_d, vector_size, cudaMemcpyDeviceToHost );

    // print results for vectors up to length 100
    if ( n <= 100 )
    {
        for ( int i = 0; i < 16; i++ )
            for ( int j = 0 ; j< 16 ; j++)
        {          
            {
                c[i][j] = a[i][j] + b[i][j];
                printf("%16.2f\n",c[i][j]);
            }
            
        }
    }

    // cleanup and quit
    cudaFree( a_d );
    cudaFree( b_d );
    cudaFree( c_d );
    free( a );
    free( b );
    free( c );
  
    return 0;
}
