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

## Reference

* [Mixed-Precision Programming with CUDA 8](https://devblogs.nvidia.com/parallelforall/mixed-precision-programming-cuda-8/)
* [parallel-forall haxpy](https://github.com/parallel-forall/code-samples/tree/master/posts/mixed-precision)
* [cublasSgemmBatched Example](https://github.com/pyrovski/cublasSgemmBatched-example)



