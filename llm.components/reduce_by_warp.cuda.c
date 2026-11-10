

__device__ __forceinline__
float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(
            0xffffffff, 
            value, 
            offset
        );
    }
    return value;
}

__global__ void reduce_warp(
    const float* input,
    float* partial,
    int N
) {
    // One value per warp
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    int lane = tid % 32;
    int warp_id = tid / 32;

    float value = (idx < N) ? input[idx] : 0.0f;

    // Reduce within each warp
    value = warp_reduce_sum(value);

    if (lane == 0) {
        warp_sums[warp_id] = value;
    }

    __syncthreads();

    int num_warps = blockDim.x / 32;

    // First warp reduces the warp partial sums
    // e.g. asuume we need two warps. warp:0 still has 32 threads running, only thread-0 and thread-1 has the partial sum:warp_sums[0:2].
    if (warp_id == 0) {
        value = (lane < num_warps)
              ? warp_sums[lane]
              : 0.0f;

        value = warp_reduce_sum(value);

        if (lane == 0) {
            partial[blockIdx.x] = value;
        }
    }
}
