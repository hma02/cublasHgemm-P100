# nvcc -o test-fp16 test-fp16.cu -arch=sm_61
make
cuda-memcheck ./hgemm

