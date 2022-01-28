# CudaVanilaTests
In this project, the summation of vector elements is performed on the CPU and GPU.  
Input: 10 vectors from 1024 to 1048576 representing powers of two from 10 to 20.  
Required libraries: CUDA Toolkit, GNUplot  
Execution: in command line mode with the command: nvcc vector_sum.cu -run   
Test system: Ubuntu 14  

Parallelization begins with simultaneous writing of two vector cells from global memory to shared memory. The addition by reduction inside the blocks continues. The addition of the results of an intra-block addition is performed by running the kernel again with the results of the previous run.
Attached are screenshots of the console and graphical output of the results.  

The console output contains information about the number of array elements, the magnitude of the calculation acceleration on the GPU, the deviation of the calculation results on the CPU and GPU in percentage terms.  

The graphical output contains the dependence of the execution time on the number of elements, in normal and logarithmic display.
It also contains a graph of the acceleration of the GPU relative to the CPU as a percentage.   
