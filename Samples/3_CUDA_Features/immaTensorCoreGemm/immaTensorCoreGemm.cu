/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a integer GEMM computation using the Warp Matrix
// Multiply and Accumulate API.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

// GEMM configuration.

#if CPU_DEBUG
#define M_TILES 16
#define N_TILES 16
#define K_TILES 8
#else
#define M_TILES 512
#define N_TILES 512
#define K_TILES 512
#endif

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

typedef int4 copy_t;

// Implementation constants.

#define WARPS_PER_BLOCK   8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 uint8_t-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 8
#else
#define CHUNK_K 16
#endif

#define CHUNK_LINE_BYTES          (CHUNK_K * K * sizeof(uint8_t))
#define WARP_COPY_BYTES           (WARP_SIZE * sizeof(copy_t))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES     (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, 
// the warp must load the matrix data using the nvcuda::wmma::load_matrix_sync operation. 
// Although the memory access pattern is not specified for that function, 
// each lane in the warp can read one or multiple matrix elements from different matrix rows or columns.
//
// For shared memory, such access can result in bank conflicts if different rows / columns of the matrix map to the same bank. 
// By shifting each row and column by a few bytes, we make sure that they map to different banks, 
// thus reducing the number of possible bank conflicts.
//
// The number of 32 one-byte "uint8" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_UINT8 32

#define checkKernelErrors(expr)                                                               \
    do {                                                                                      \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess) {                                                           \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

enum kernels {
    shmem_uint8_wmma_gemm            = 0, // uint8 WarpMMA with shmem.
    simple_uint8_wmma_gemm           = 1  // uint8 WarpMMA w/o shmem.
};
const char *kernelNames[] = {"shmem_uint8_wmma_gemm", "simple_uint8_wmma_gemm"};

using namespace nvcuda;

__host__ void init_host_matrices(uint8_t *a, uint8_t *b, int *c)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = (uint8_t)(rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = (uint8_t)(rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = (rand() % 3);
    }
}

__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B, const int *C, int *D, int alpha, int beta)
{
    extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8]; // skew 32 uint8 to avoid bank conflict and align to 256-bit

    // Warp and lane identification.
    const auto numWarpsPerMat = WARPS_PER_BLOCK / 2;
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;
    const unsigned int warpGroupId = warpId / BLOCK_ROW_WARPS;
    const unsigned int warpIdInGroup = warpId % BLOCK_ROW_WARPS;
    const unsigned int warpIdInMatGroup = warpId % numWarpsPerMat;
    const auto isWarpForMatA = warpId < numWarpsPerMat;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    int *shmem_warp_tile_ptr = (int *)&shmem[0][0] 
                                + warpGroupId * SHMEM_STRIDE * N * BLOCK_ROW_WARPS
                                + warpIdInGroup * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES)
            break;

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx                 = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const int   *src_gmem_warp_stream_ptr = &C[gmem_idx];

        /********** Load C from global memory to shared memory ***********/

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES][WARP_ROW_TILES];

        /********** Load C from shared memory to fragments ***********/

// Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        /********** Scale C by beta/alpha ***********/

// Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const uint8_t *warp_ptr = isWarpForMatA ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * warpIdInMatGroup * 2)
                                               : (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * warpIdInMatGroup * 2);

        /********** Apply GEMM for each chunk of K tiles ***********/

// Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {

            /********** Load A,B from global memory to shared memory ***********/

            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx = isWarpForMatA
                                 ? (M * warpIdInMatGroup * 2)
                                 : (N * warpIdInMatGroup * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            copy_t *lane_ptr = (copy_t *)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL)
                           + (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                // Copy 16 bytes at once in each lane.
                *((copy_t *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (copy_t *)((uint8_t *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            /********** Accumulate A * B + C into C's shared memory ***********/

// Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major> b[WARP_ROW_TILES];

                /********** Load A from shared memory to fragments ***********/

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t         shmem_idx_a = warpGroupId * M * 2 + (i * M);
                    const uint8_t *tile_ptr    = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

                /********** Load B from shared memory to fragments ***********/

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b      = shmem_idx_b_off + (WARP_ROW_TILES * N) * warpIdInGroup + (j * N);
                            const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                        }

                        /********** Perform C += A * B + C ***********/

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        /********** Scale C and Store D from fragments to shared memory to global memory ***********/

// Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
        
        /********** Scale C by alpha ***********/

#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                /********** Store C from fragments to shared memory ***********/

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        /********** Store D from shared memory to global memory ***********/

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
            *((copy_t *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
}

// Performs an MxNxK GEMM (D = alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm_imma kernel. It
// is designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
__global__ void simple_wmma_gemm_imma(const uint8_t *a,
                                      const uint8_t *b,
                                      const int     *c,
                                      int           *d,
                                      int            m,
                                      int            n,
                                      int            k,
                                      int            alpha,
                                      int            beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = m;
    int ldb = k;
    int ldc = n;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, int>                   acc_frag;
    wmma::fragment<wmma::accumulator, M, N, K, int>                   c_frag;

    // zero-init the acc fragment
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k; i += K) {
        int aCol = i;
        int aRow = warpM * M;

        int bCol = i;
        int bRow = warpN * N;

        // Bounds checking
        if (aRow < m && aCol < k && bRow < k && bCol < n) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(b_frag, b + bRow * ldb + bCol, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, 
    // and add this our result scaled by alpha
    int cCol = warpN * N;
    int cRow = warpM * M;
    if (cRow < m && cCol < n) {
        wmma::load_matrix_sync(c_frag, c + cRow * ldc + cCol, ldc, C_LAYOUT);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cRow * ldc + cCol, c_frag, ldc, C_LAYOUT);
    }
}

__host__ void matMultiplyOnHost(uint8_t *A,
                                uint8_t *B,
                                int     *C,
                                int      alpha,
                                int      beta,
                                int      numARows,
                                int      numAColumns,
                                int      numBRows,
                                int      numBColumns,
                                int      numCRows,
                                int      numCColumns)
{
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            int temp = 0;

            for (int k = 0; k < numAColumns; k++) {
                temp += A[i * numAColumns + k] * B[j * numBRows + k];
            }

            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}

int main(int argc, char **argv)
{
    printf("Initializing...\n");

    int dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM72) architecture or higher.
    if (deviceProp.major < 7 || (deviceProp.major <= 7 && deviceProp.minor < 2)) {
        printf("immaTensorCoreGemm requires SM 7.2 or higher to use Tensor Cores.  "
               "Exiting...\n");
        exit(EXIT_WAIVED);
    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    uint8_t *A_h = NULL;
    uint8_t *B_h = NULL;
    int     *C_h = NULL;
#if CPU_DEBUG
    int *result_hD   = NULL;
    int *result_host = NULL;
#endif

    A_h = (uint8_t *)malloc(sizeof(uint8_t) * M_GLOBAL * K_GLOBAL);
    B_h = (uint8_t *)malloc(sizeof(uint8_t) * K_GLOBAL * N_GLOBAL);
    C_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
    result_hD   = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
    result_host = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#endif

    uint8_t *A = NULL;
    uint8_t *B = NULL;
    int     *C = NULL;
    int     *D = NULL;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), sizeof(uint8_t) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), sizeof(uint8_t) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), sizeof(int) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D), sizeof(int) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);
    assert(BLOCK_ROW_WARPS * BLOCK_COL_WARPS == WARPS_PER_BLOCK);
    assert(M_TILES >= 1 && N_TILES >= 8 && K_TILES >= 8);

    init_host_matrices(A_h, B_h, C_h);

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(uint8_t) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(uint8_t) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(int) * M_GLOBAL * N_GLOBAL));

    printf("Preparing data for GPU...\n");

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    enum {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
        // per-CTA chunks
        // of the A and B matrices. Therefore, the right amount to request is the
        // maximum of those
        // two numbers.
        SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
    };

    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

    int alpha = 1;
    int beta  = 1;

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // kernel to run - default (shmem_uint8_wmma_gemm == 0)
    kernels selected_kernel = shmem_uint8_wmma_gemm;

    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        int kernel_number = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
        if (kernel_number < 2) {
            selected_kernel = (kernels)kernel_number;
        }
        else {
            printf("Error: kernel number should be between 0 to 1, you have entered %d\n", kernel_number);
            exit(EXIT_FAILURE);
        }
    }

    // If enough shared memory available on the GPU use high performant kernel
    if ((deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) && (selected_kernel != simple_uint8_wmma_gemm)) {
        printf("Computing using high performance kernel = %d - %s\n", selected_kernel, kernelNames[selected_kernel]);

        checkCudaErrors(cudaFuncSetAttribute(compute_gemm_imma, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
        checkKernelErrors((compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(
            A, B, C, D, alpha, beta)));
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
#endif
    }
    else {
        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multiple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        auto block_num_tiles_m = M * blockDim.x / WARP_SIZE;
        auto block_num_tiles_n = N * blockDim.y;

        gridDim.x = (M_GLOBAL + block_num_tiles_m - 1) / block_num_tiles_m;
        gridDim.y = (N_GLOBAL + block_num_tiles_n - 1) / block_num_tiles_n;

        printf("Computing... using simple_wmma_gemm_imma kernel\n");
        simple_wmma_gemm_imma<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
#endif
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Timing to calculate TOPS
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time: %f ms\n", milliseconds);
    printf("TOPS (uint8): %.4f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);

#if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");

    memcpy(result_host, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL);

    matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

    auto passed = true;
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            auto idx = i * N_GLOBAL + j;
            if (fabs(result_hD[idx] - result_host[idx]) > 0.1f) {
                printf("mismatch (i=%d, j=%d) result_hD=%d result_host=%d\n", i, j, result_hD[idx], result_host[idx]);
                passed = false;
            }   
        }
    }
    printf("Verification %s\n", passed ? "PASSED" : "FAILED");
    free(result_host);
    free(result_hD);
#endif

    free(A_h);
    free(B_h);
    free(C_h);
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

    return EXIT_SUCCESS;
}
