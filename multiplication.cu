#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void multiplication(int* a, int* b, int* c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] * b[i];
}

__managed__ int a[256], b[256], c[256];

// name of main function changed (see one.cu)

int two_main() {
    for (int i = 0; i < 256; i++) {
        a[i] = i;
        b[i] = i * (i - (i - 5));
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, 256 * sizeof(int));
    cudaMalloc(&d_b, 256 * sizeof(int));
    cudaMalloc(&d_c, 256 * sizeof(int));

    cudaMemcpy(d_a, a, 256 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 256 * sizeof(int), cudaMemcpyHostToDevice);

    multiplication << <1, 256 >> > (d_a, d_b, d_c);

    cudaMemcpy(c, d_c, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 256; i++) {
        printf("%d * %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
