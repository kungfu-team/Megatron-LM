#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(e)                                                               \
    if (e != cudaSuccess) {                                                    \
        exit(1);                                                               \
    }

void show_cuda_rt()
{
    int runtimeVersion;
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION
    cudaError_t ret = cudaRuntimeGetVersion(&runtimeVersion);
    CHECK(ret);
    printf("%d\n", runtimeVersion);
    int major = runtimeVersion / 1000;
    int minor = runtimeVersion % 1000 / 10;
    printf("%d.%d\n", major, minor);
}

int main()
{
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE
    cudaInitDevice(0, 0, 0);
    show_cuda_rt();
    return 0;
}
