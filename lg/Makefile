CUDA_HOME=/usr/local/cuda

NVCC=$(CUDA_HOME)/bin/nvcc

all: \
	check-cuda-driver \
	check-cuda-rt \
	check

check-cuda-driver:
	$(NVCC) check-cuda-driver.c -L $(CUDA_HOME)/lib64 -lcuda -o check-cuda-driver.out

check-cuda-rt:
	$(NVCC) check-cuda-rt.c -L $(CUDA_HOME)/lib64 -lcudart -o check-cuda-rt.out

check:
	./check-cuda-driver.out
	./check-cuda-rt.out
