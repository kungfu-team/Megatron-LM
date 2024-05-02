//  /usr/local/cuda/bin/nvcc check.c -L /usr/local/cuda/lib64 -lcuda

#include <cuda.h>
#include <stdio.h>

void check_result(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        exit(1);
    }
}

int main()
{
    CUresult result;

    int flags = 0;
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE
    result = cuInit(flags);
    check_result(result);

    int device = 0;

    int comp_cap_major = 0;
    int comp_cap_minor = 0;
    result = cuDeviceGetAttribute(
        &comp_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    check_result(result);

    result = cuDeviceGetAttribute(
        &comp_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    check_result(result);

    printf("%d.%d\n", comp_cap_major, comp_cap_minor);
    return 0;
}

// https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
