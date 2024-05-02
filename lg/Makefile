CUDA_HOME=/usr/local/cuda

NVCC=$(CUDA_HOME)/bin/nvcc

main:
	$(NVCC) check.c -L $(CUDA_HOME)/lib64 -lcuda
