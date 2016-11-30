# fp16-cublasHgemm-test
A simple benchmarking code of the float16 performance on Tesla P100 GPU (sm_60) based on cublasHgemm.

##Build and Run

The code does `C=alpha*A*B+beta*C` on GPU with different sizes of square matrices A, B and C. Shape A is (m,k). Shape B is (k,n). Shape C is (m,n).

To test float16 matrix multiplication,

```shell
$ make
$ ./hgemm
```

Uncomment line 11 in `hgemm.cu` to test float32 matrix multiplication.

## Example Testing Result

```shell
nvcc hgemm.cu -lcublas --std=c++11 -arch=sm_60  -o hgemm

running cublasHgemm test

running with min_m_k_n: 2 max_m_k_n: 32768 repeats: 10
allocating device variables
float16; size 2 average: 2.48544e-05 s 
float16; size 4 average: 1.42016e-05 s 
float16; size 8 average: 1.94624e-05 s 
float16; size 16 average: 2.21248e-05 s 
float16; size 32 average: 2.50496e-05 s 
float16; size 64 average: 3.10016e-05 s 
float16; size 128 average: 3.09376e-05 s 
float16; size 256 average: 3.93728e-05 s 
float16; size 512 average: 6.07232e-05 s 
float16; size 1024 average: 0.000199133 s 
float16; size 2048 average: 0.00114664 s 
float16; size 4096 average: 0.00819163 s 
float16; size 8192 average: 0.0627873 s 
float16; size 16384 average: 0.494198 s 
float16; size 32768 average: 3.96341 s 


nvcc hgemm.cu -lcublas --std=c++11 -arch=sm_60  -o hgemm

running cublasSgemm test

running with min_m_k_n: 2 max_m_k_n: 32768 repeats: 10
allocating device variables
float32; size 2 average: 1.6112e-05 s 
float32; size 4 average: 1.89216e-05 s 
float32; size 8 average: 1.84352e-05 s 
float32; size 16 average: 5.792e-06 s 
float32; size 32 average: 7.8752e-06 s 
float32; size 64 average: 8.2336e-06 s 
float32; size 128 average: 2.37376e-05 s 
float32; size 256 average: 3.65504e-05 s 
float32; size 512 average: 8.088e-05 s 
float32; size 1024 average: 0.000374634 s 
float32; size 2048 average: 0.0019914 s 
float32; size 4096 average: 0.016146 s 
float32; size 8192 average: 0.122788 s 
float32; size 16384 average: 0.980065 s 
float32; size 32768 average: 7.92793 s 
```

## Reference

* [Mixed-Precision Programming with CUDA 8](https://devblogs.nvidia.com/parallelforall/mixed-precision-programming-cuda-8/)
* [parallel-forall haxpy](https://github.com/parallel-forall/code-samples/tree/master/posts/mixed-precision)
* [cublasSgemmBatched Example](https://github.com/pyrovski/cublasSgemmBatched-example)



