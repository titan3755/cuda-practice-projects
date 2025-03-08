#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <math.h>

__global__ void sieve_segment(uint64_t start, uint64_t end, uint64_t* primes, uint64_t numPrimes, uint64_t* segment) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num = start + index;

    if (num < end) {
        segment[index] = 1;  // Assume the number is prime
        for (uint64_t i = 0; i < numPrimes; i++) {
            if (num % primes[i] == 0) {
                segment[index] = 0;  // Mark as non-prime
                break;
            }
        }
    }
}

int main(void) {
    uint64_t N = static_cast<uint64_t>(1 << 30); // Set N to 2^25
    uint64_t segmentSize = 1 << 10; // Process in segments of 2^20
    uint64_t* primes;
    uint64_t* segment;

    // Allocate memory for primes and segment
    cudaMallocManaged(&primes, (sqrt(N) + 1) * sizeof(uint64_t));
    cudaMallocManaged(&segment, segmentSize * sizeof(uint64_t));

    // Initialize primes array (basic sieve up to sqrt(N))
    for (uint64_t i = 0; i <= sqrt(N); i++) {
        primes[i] = 1;
    }
    primes[0] = primes[1] = 0;
    for (uint64_t i = 2; i * i <= sqrt(N); i++) {
        if (primes[i]) {
            for (uint64_t j = i * i; j <= sqrt(N); j += i) {
                primes[j] = 0;
            }
        }
    }

    // Extract list of primes up to sqrt(N)
    uint64_t numPrimes = 0;
    for (uint64_t i = 2; i <= sqrt(N); i++) {
        if (primes[i]) {
            primes[numPrimes++] = i;
        }
    }

    // Process the range [0, N) in segments
    for (uint64_t start = 0; start < N; start += segmentSize) {
        uint64_t end = start + segmentSize;
        if (end > N) {
            end = N;
        }

        // Launch the segmented sieve kernel
        uint64_t blockSize = 256;
        uint64_t numBlocks = (segmentSize + blockSize - 1) / blockSize;
        sieve_segment << <numBlocks, blockSize >> > (start, end, primes, numPrimes, segment);

        // Synchronize the device to ensure kernel execution finishes
        cudaDeviceSynchronize();

        // Print primes in the current segment
        for (uint64_t i = 0; i < segmentSize && (start + i) < N; i++) {
            if (segment[i]) {
                printf("%lu\n", start + i);
            }
        }
    }

    // Free the allocated memory
    cudaFree(primes);
    cudaFree(segment);

    return 0;
}