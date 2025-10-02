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

#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>

#include "scan_common.h"

// All three kernels run 512 threads per workgroup
// Must be a power of two
#define THREADBLOCK_SIZE 256


////////////////////////////////////////////////////////////////////////////////
//  Two-Phase Inclusive Prefix Sum Kernel (Blelloch scan algorithm)
////////////////////////////////////////////////////////////////////////////////
// This kernel assumes blockDim.x is a power of 2.
// It processes one block of elements, handling potential out-of-bounds access
// for the last block if N is not a multiple of blockDim.x.
// and this kernel's "Work Complexity" is O(N), "Parallel Time Complexity" is O(logN)
// thus it is a work-optimal algorithm, since the non-parallel work is still O(N)
__global__ void inclusive_prefix_sum_kernel(int* d_input, int* d_output, int N) {
    // Dynamically allocated shared memory; size is specified at kernel launch via the 3rd parameter
    extern __shared__ int s_data[]; 

    int tid = threadIdx.x;                      // Thread index inside the block
    int block_start_idx = blockIdx.x * blockDim.x; // Global start index for this block
    int global_idx = block_start_idx + tid;     // Global index for the current thread

    // Load data from global memory to shared memory
    // Also keep the original value for the final inclusive conversion
    int original_val = 0; 
    if (global_idx < N) {
        original_val = d_input[global_idx];
    }
    s_data[tid] = original_val;                 // Store original value in shared memory
    __syncthreads();                            // Ensure all threads have loaded their data

    // Phase 1: Up-Sweep (Reduction)
    // Loop variable s is the current stride (1, 2, 4, ...)
    for (unsigned int s = 1; s < blockDim.x; s <<= 1) { 
        __syncthreads();                        // Wait for the previous step to finish
        
        // Only threads meeting the condition perform the update:
        // tid must be >= s (left element exists) and must be the rightmost element
        // of the current segment of length 2*s (tid % (2*s) == 2*s - 1)
        if ((tid >= s) && ((tid % (2 * s)) == (2 * s - 1))) {
            s_data[tid] += s_data[tid - s];     // Add left element to right element
        }
    }
    // After this phase, s_data[blockDim.x - 1] contains the total sum of the block.
    // Other elements store intermediate sums.

    // Phase 2: Down-Sweep (Scan)

    // 1. Set the last element to 0 to prepare for exclusive scan
    if (tid == blockDim.x - 1) {                // Only the last thread in the block
        s_data[blockDim.x - 1] = 0;             // Zero the total to turn the reduction into exclusive scan
    }
    __syncthreads();                            // Ensure the last element has been zeroed

    // 2. Down-sweep loop
    // Loop variable s runs backwards from blockDim.x/2 down to 1 (4, 2, 1, ...)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { 
        __syncthreads();                        // Synchronize every step
        
        // Only threads meeting the condition perform the update:
        // tid must be >= s and must be the rightmost element of the current segment
        if ((tid >= s) && ((tid % (2 * s)) == (2 * s - 1))) {
            int val_left_child = s_data[tid - s]; // Save original value of left child
            s_data[tid - s] = s_data[tid];        // Left child gets parent's value (its right sibling)
            s_data[tid] += val_left_child;        // Right child adds the original left-child value
        }
    }
    // After this phase, s_data contains the exclusive prefix sum for the block.
    // Example: [0, a, a+b, a+b+c, ...]

    // Final step: convert exclusive prefix sum to inclusive and write to global memory
    // inclusive_sum[i] = exclusive_sum[i] + original_input[i]
    if (global_idx < N) {
        d_output[global_idx] = s_data[tid] + original_val;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets (Hillis-Steele scan algorithm)
////////////////////////////////////////////////////////////////////////////////
// Naive inclusive scan:
// Allocate 2 * 'size' local memory, initialize the first half
// with 'size' zeros avoiding if(pos >= offset) condition evaluation and saving instructions
// and this kernel's "Work Complexity" is O(NlogN), "Parallel Time Complexity" is O(logN)
// thus it is not a work-optimal algorithm, since the non-parallel work is O(N), and it offers logN times of work
// though it has the same optimal parallel time complexity
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    uint pos    = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        cg::sync(cta);
        uint t = s_Data[pos] + s_Data[pos - offset];
        cg::sync(cta);
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    return scan1Inclusive(idata, s_Data, size, cta) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    // Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    // Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

// Exclusive vector scan: the array to be scanned is stored in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    uint4 idata4 = d_Src[pos];

    // Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);

    // Write back
    d_Dst[pos] = odata4;
}

// Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(uint *d_Buf, uint *d_Dst, uint *d_Src, uint N, uint arrayLength)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  s_Data[2 * THREADBLOCK_SIZE];

    // Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load top elements
    // Convert results of bottom-level scan back to inclusive
    uint idata = 0;

    if (pos < N)
        idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos]
              + d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    // Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength, cta);

    // Avoid out-of-bound access
    if (pos < N) {
        d_Buf[pos] = odata;
    }
}

// Final step of large-array scan: combine basic inclusive scan with exclusive
// scan of top elements of input arrays
__global__ void uniformUpdate(uint4 *d_Data, uint *d_Buffer)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  buf;
    uint             pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        buf = d_Buffer[blockIdx.x];
    }

    cg::sync(cta);

    uint4 data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
// Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
// Due to scanExclusiveShared<<<>>>() 1D block addressing
extern "C" const uint MAX_BATCH_ELEMENTS   = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;

// Internal exclusive scan buffer
static uint *d_Buf;

extern "C" void initScan(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(uint)));
}

extern "C" void closeScan(void) { checkCudaErrors(cudaFree(d_Buf)); }

// `factorRadix2` will factorize L into L = 2^power_factor * odd_factor
// e.g. L = 12 = 2^2 * 3; L = 7 = 2^0 * 7; L = 16 = 2^4 * 1
// where power_factor will be store into `log2L` and odd_factor will be returned
static inline uint factorRadix2(uint &log2L, uint L)
{
    if (!L) {
        log2L = 0;
        return 0;
    }
    else {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++)
            ;

        return L;
    }
}

static inline bool isPowerOf2(uint x) { return ((x != 0) && ((x & (x - 1)) == 0)); }

static inline uint iDivUp(uint dividend, uint divisor)
{
    // return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
    // The above way is not the most efficient and elegant way
    return (dividend + divisor - 1) / divisor;
}

extern "C" size_t scanExclusiveShort(uint *d_Dst, uint *d_Src, uint batchSize, uint arrayLength)
{
    // Check power-of-two factorization
    // uint log2L;
    // uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    // assert(factorizationRemainder == 1); // then arrayLength is a power of 2
    // The above way is not the most efficient and elegant way
    assert(isPowerOf2(arrayLength));

    // Check supported size range
    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

    // Check total batch size limit
    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    // Check all threadblocks to be fully packed with data
    assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);

    // vectorize the uint data to uint4
    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst, (uint4 *)d_Src, arrayLength);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    return THREADBLOCK_SIZE;
}

// The following function is for large array
// whose array size > 4 * THREADBLOCK_SIZE
extern "C" size_t scanExclusiveLarge(uint *d_Dst, uint *d_Src, uint batchSize, uint arrayLength)
{
    // Check power-of-two factorization
    // uint log2L;
    // uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    // assert(factorizationRemainder == 1); // then arrayLength is a power of 2
    // The above way is not the most efficient and elegant way
    assert(isPowerOf2(arrayLength));

    // Check supported size range
    assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));

    // Check total batch size limit
    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst, (uint4 *)d_Src, 4 * THREADBLOCK_SIZE);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    // Not all threadblocks need to be packed with input data:
    // inactive threads of highest threadblock just don't do global reads and writes
    const uint blockCount2 = iDivUp((batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
    scanExclusiveShared2<<<blockCount2, THREADBLOCK_SIZE>>>((uint *)d_Buf,
                                                            (uint *)d_Dst,
                                                            (uint *)d_Src,
                                                            (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                                                            arrayLength / (4 * THREADBLOCK_SIZE));
    getLastCudaError("scanExclusiveShared2() execution FAILED\n");

    uniformUpdate<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>((uint4 *)d_Dst,
                                                                                            (uint *)d_Buf);
    getLastCudaError("uniformUpdate() execution FAILED\n");

    return THREADBLOCK_SIZE;
}
