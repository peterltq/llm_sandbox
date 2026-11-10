
__global__ void reduce_shared(
    const float* input,
    float* partial,
    int N
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;

    __syncthreads();

    // Tree reduction
    for (unsigned int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // One partial sum per block
    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

////// Launch configurationb
int threads = 256;
int blocks = (N + threads - 1) / threads;

float* d_partial;
cudaMalloc(&d_partial, blocks * sizeof(float));

reduce_shared<<<
    blocks,
    threads,
    threads * sizeof(float)
>>>(d_input, d_partial, N);