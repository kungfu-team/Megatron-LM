CUDA_HOME=/usr/local/cuda

NVCC=$(CUDA_HOME)/bin/nvcc

all: \
	check-cuda-driver \
	check-cuda-rt \
	gen-torch-requirements \
	check

check-cuda-driver:
	$(NVCC) check-cuda-driver.c -L $(CUDA_HOME)/lib64 -lcuda -o check-cuda-driver.out

check-cuda-rt:
	$(NVCC) check-cuda-rt.c -L $(CUDA_HOME)/lib64 -lcudart -o check-cuda-rt.out

gen-torch-requirements:
	$(NVCC) gen-torch-requirements.c -L $(CUDA_HOME)/lib64 -lcudart -o gen-torch-requirements.out

check:
	./check-cuda-driver.out
	./check-cuda-rt.out
