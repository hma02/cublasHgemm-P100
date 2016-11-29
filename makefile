CUDA_ARCH_FLAGS := -arch=sm_60
CC_FLAGS += -lcublas
# CC_FLAGS += -lcurand
# CC_FLAGS += -Xptxas
# CC_FLAGS += -v
# CC_FLAGS += -O3
CC_FLAGS += --std=c++11 $(CUDA_ARCH_FLAGS)

EXE = hgemm

all: $(EXE)

% : %.cu
	nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean:
	rm -f $(EXE)
