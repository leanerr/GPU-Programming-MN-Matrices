
**Assignment III**

**GPU Programming – Fall 2021**

1. Change into the 08-intro-to-cuda directory.
1. Examine the source code in add-vectors.cu until you are comfortable with its operation. In particular, be sure you can identify which parts of the program correspond with each part of the pattern described in the program's heading comments.
1. Compile and run the program: 

nvcc -o add-vectors add-vectors.cu

./add-vectors

1. The output will probably not be too exciting but should convince you the program is working correctly. Try running the program with different vector lengths 

./add-vectors 5

./add-vectors 50

./add-vectors 10000

./add-vectors 100000000

The program doesn't display vectors longer than 100 elements, so the last two commands won't produce any output. Notice, however, that the computation is correct for a range of sizes, even though our block size was set to 16.

1. CUDA SDKs since version 5.0 have included a profiler. You do not need to instrument and/or recompile your code; just run the profiler with your program and any arguments: 

nvprof ./add-vectors 1000

The output will timing information for each CUDA function. Notice that the program spends most of its time allocating memory on the device when the vector length is 1000. Now try 

nvprof ./add-vectors  100000000

and you should find very different behavior; the time to copy memory to and from the device is the dominant time.

**Now it's your turn**

**Exercise:** Write a program that initializes two *M*×*N* matrices and computes the sum of the two matrices on the GPU device. After copying the result back to the host, your program should print out the result matrix if *N*≤10. You may use add-vectors.cu as a starting point or start from scratch.

It is natural to use a 2D grid for a matrix. In this case the block\_size and num\_blocks variables should be of type dim3. The kernel launch area show below accomplishes this 

`  `dim3 block\_size( 16, 16 );

`  `dim3 num\_blocks( ( n - 1 + block\_size.x ) / block\_size.x, 

`                   `( m - 1 + block\_size.y ) / block\_size.y );

`  `add\_matrices<<< num\_blocks, block\_size >>>( c\_d, a\_d, b\_d, m, n );

Of course, the kernel code will need to work correctly with a 2D grid rather than the 1D grid used in add-vectors.cu. 

Test your code with a range of values of *M* and *N*. For each case, run your program both without and with the profiler. 

**What to turn in**

Please turn in a printout of your final matrix-addition source code along with a short report summarizing the profiling data.

