#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                            \
do {                                                                \
    cudaError_t error = (call);                                     \
    if (error != cudaSuccess) {                                     \
        std::cerr << "CUDA error: "                                 \
                  << cudaGetErrorString(error)                       \
                  << " at " << __FILE__ << ":" << __LINE__          \
                  << "\n";                                          \
        std::exit(EXIT_FAILURE);                                    \
    }                                                               \
} while (0)

constexpr int TILE_SIZE = 16;

/*
 * Matrix layouts are row-major:
 *
 * A: M x K
 * B: K x N
 * C: M x N
 */
__global__ void tiledMatMul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numberOfTiles =
        (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numberOfTiles; ++tile) {
        int aColumn = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        /*
         * Cooperatively load one tile of A.
         *
         * Neighboring threads vary threadIdx.x, so they read
         * neighboring A elements. This produces coalesced reads.
         */
        if (row < M && aColumn < K) {
            tileA[threadIdx.y][threadIdx.x] =
                A[row * K + aColumn];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        /*
         * Cooperatively load one tile of B.
         *
         * Neighboring threads also read neighboring B columns,
         * producing coalesced reads.
         */
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Wait until the complete A and B tiles are loaded.
        __syncthreads();

        /*
         * Every thread calculates one C element.
         *
         * The values are read from shared memory rather than
         * repeatedly fetched from global memory.
         */
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k]
                 * tileB[k][threadIdx.x];
        }

        /*
         * All threads must finish using these shared-memory tiles
         * before the next iteration overwrites them.
         */
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Kernel launcher
void launchTiledMatMul(
    const float* deviceA,
    const float* deviceB,
    float* deviceC,
    int M,
    int N,
    int K)
{
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    tiledMatMul<<<grid, block>>>(
        deviceA,
        deviceB,
        deviceC,
        M,
        N,
        K
    );

    CHECK_CUDA(cudaGetLastError());
}

// Test program
int main()
{
    const int M = 513;
    const int K = 257;
    const int N = 769;

    std::size_t sizeA =
        static_cast<std::size_t>(M) * K;

    std::size_t sizeB =
        static_cast<std::size_t>(K) * N;

    std::size_t sizeC =
        static_cast<std::size_t>(M) * N;

    float* hostA = new float[sizeA];
    float* hostB = new float[sizeB];
    float* hostC = new float[sizeC];

    for (std::size_t i = 0; i < sizeA; ++i) {
        hostA[i] = static_cast<float>((i % 7) - 3);
    }

    for (std::size_t i = 0; i < sizeB; ++i) {
        hostB[i] = static_cast<float>((i % 5) - 2);
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    CHECK_CUDA(cudaMalloc(
        &deviceA,
        sizeA * sizeof(float)
    ));

    CHECK_CUDA(cudaMalloc(
        &deviceB,
        sizeB * sizeof(float)
    ));

    CHECK_CUDA(cudaMalloc(
        &deviceC,
        sizeC * sizeof(float)
    ));

    CHECK_CUDA(cudaMemcpy(
        deviceA,
        hostA,
        sizeA * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        deviceB,
        hostB,
        sizeB * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    launchTiledMatMul(
        deviceA,
        deviceB,
        deviceC,
        M,
        N,
        K
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        hostC,
        deviceC,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    /*
     * Verify a few output elements using CPU calculation.
     */
    bool correct = true;

    for (int row = 0; row < M && correct; row += 101) {
        for (int col = 0; col < N; col += 137) {
            float expected = 0.0f;

            for (int k = 0; k < K; ++k) {
                expected +=
                    hostA[row * K + k] *
                    hostB[k * N + col];
            }

            float actual = hostC[row * N + col];

            if (std::fabs(expected - actual) > 1e-3f) {
                std::cerr
                    << "Mismatch at (" << row << ", "
                    << col << "): expected " << expected
                    << ", actual " << actual << "\n";

                correct = false;
                break;
            }
        }
    }

    std::cout << (correct ? "Correct\n" : "Incorrect\n");

    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceC));

    delete[] hostA;
    delete[] hostB;
    delete[] hostC;

    return correct ? 0 : 1;
}