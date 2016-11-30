# fp16-cublasHgemm-test
A simple benchmarking code of the float16 performance on Tesla P100 GPU (sm_60) based on cublasHgemm.

##Build and Run

The code does `C=alpha*A*B+beta*C` on GPU with different sizes of square matrices A, B and C. Shape A is (m,k). Shape B is (k,n). Shape C is (m,n).

To test float16 matrix multiplication,

```shell
$ make
$ ./hgemm
```

Comment line 11 in `hgemm.cu` to test float32 matrix multiplication.

## Example Testing  Result

```shell
nvcc hgemm.cu -lcublas --std=c++11 -arch=sm_60  -o hgemm

running cublasHgemm test

running with min_m_k_n: 2 max_m_k_n: 32768 repeats: 10
allocating device variables
float16; size 2 average: 7.69632e-05 s 
float16; size 4 average: 1.34304e-05 s 
float16; size 8 average: 3.49152e-05 s 
float16; size 16 average: 1.6272e-05 s 
float16; size 32 average: 1.91808e-05 s 
float16; size 64 average: 2.52672e-05 s 
float16; size 128 average: 2.48512e-05 s 
float16; size 256 average: 6.52992e-05 s 
float16; size 512 average: 0.000111104 s 
float16; size 1024 average: 0.000275123 s 
float16; size 2048 average: 0.00155046 s 
float16; size 4096 average: 0.00934949 s 
float16; size 8192 average: 0.0659167 s 
float16; size 16384 average: 0.508014 s 
float16; size 32768 average: 4.01786 s 

nvcc hgemm.cu -lcublas --std=c++11 -arch=sm_60  -o hgemm

running cublasSgemm test

running with min_m_k_n: 2 max_m_k_n: 32768 repeats: 10
allocating device variables
float32; size 2 average: 5.21152e-05 s 
float32; size 4 average: 2.06112e-05 s 
float32; size 8 average: 7.1616e-06 s 
float32; size 16 average: 5.3248e-06 s 
float32; size 32 average: 4.624e-06 s 
float32; size 64 average: 1.128e-05 s 
float32; size 128 average: 2.37504e-05 s 
float32; size 256 average: 4.83776e-05 s 
float32; size 512 average: 0.000117616 s 
float32; size 1024 average: 0.000599805 s 
float32; size 2048 average: 0.0026987 s 
float32; size 4096 average: 0.0180615 s 
float32; size 8192 average: 0.128823 s 
float32; size 16384 average: 1.00408 s 
float32; size 32768 average: 8.07247 s 
```
## Reference

* [Mixed-Precision Programming with CUDA 8](https://devblogs.nvidia.com/parallelforall/mixed-precision-programming-cuda-8/)
* [parallel-forall haxpy](https://github.com/parallel-forall/code-samples/tree/master/posts/mixed-precision)
* [cublasSgemmBatched Example](https://github.com/pyrovski/cublasSgemmBatched-example)



