#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include "utils.h"       
#include "vec_dtypes.cuh" 


const float FP8_E4M3_MIN = -448.0f;
constexpr uint16_t dst_bias = 127;
constexpr uint16_t dst_0p5 = 16128;   
constexpr uint16_t dst_m_bits = 7;

__device__ __forceinline__ float decode_mxfp4_scale(uint8_t ue8m0_value) {

    uint16_t bf16_bits = static_cast<uint16_t>(ue8m0_value) << 7;
    return __bfloat162float(__ushort_as_bfloat16(bf16_bits));
}


__device__ __forceinline__ float compute_te_pow2_scale(float amax, float eps) {
    if (amax <= eps) {
        return 1.0f; 
    }

    
    float raw_scale = amax / FP8_E4M3_MAX;

    uint32_t s_bits = __float_as_uint(raw_scale);

    uint32_t scale_exp = (s_bits >> 23) & 0xFF;

    if ((s_bits & 0x7FFFFF) != 0) {
        scale_exp += 1;
    }


    float aligned_divisor = __uint_as_float(scale_exp << 23);

    return 1.0 / aligned_divisor;
}

// ========================================================================
// Kernel
// ========================================================================
template <
    typename T,
    typename DST_DTYPE = __nv_fp8_e4m3>
__global__ void omo_per_token_Mxfp4RowToBf16ToFp8Col_split_kernel(
    const uint8_t* __restrict__ input,          // [Total_M, N/2] Packed MXFP4
    const uint8_t* __restrict__ input_s,        // [Total_M, N/32] Packed Scale
    DST_DTYPE* __restrict__ output_q,           // [N, Total_M]  <-- Transposed Output
    float* __restrict__ output_s,               // [N, Sum_Scale_Blocks] (Stores 1/Scale)
    const int* __restrict__ split_lens,         // [Num_Splits]
    const int* __restrict__ split_offsets,      // [Num_Splits]
    const int* __restrict__ scale_offsets,      // [Num_Splits]
    const int N,
    const int Total_M,
    const float eps) {

    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int THREADS_PER_BLOCK = 1024;
    constexpr int ELEMENTS_PER_THREAD = (TILE_M * TILE_N) / THREADS_PER_BLOCK; // 16
    constexpr int VEC_SIZE = 16; 

    // Grid Coordinates
    const int split_idx = blockIdx.x;
    const int block_m_in_split = blockIdx.y;
    const int block_col = blockIdx.z;
    const int tid = threadIdx.x;

    // Metadata & Early Exit
    const int current_split_len = split_lens[split_idx];
    if (block_m_in_split * TILE_M >= current_split_len) return;

    const int current_split_start_row = split_offsets[split_idx];
    const int current_scale_start_row = scale_offsets[split_idx];

    // Shared Memory
    extern __shared__ float smem[];
    float (*smem_dequant)[TILE_N + 1] = (float (*)[TILE_N + 1])smem;
    float* smem_scale = smem + TILE_M * (TILE_N + 1);

    // Global Coordinates
    const int global_row_base = current_split_start_row + block_m_in_split * TILE_M;
    const int global_col_start = block_col * TILE_N;
    const int valid_rows_in_tile = min(TILE_M, current_split_len - block_m_in_split * TILE_M);

    using vec_t_u8_8 = omo::vec_t<uint8_t, 8>;
    
    // --- Step 1: Load, Decode (MXFP4->Float), Transpose Store ---
    int linear_idx_start = tid * ELEMENTS_PER_THREAD;
    int local_row = linear_idx_start / TILE_N;
    int local_col_start = linear_idx_start % TILE_N;

    if (local_row < TILE_M && local_col_start < TILE_N) {
        int global_row = global_row_base + local_row;
        int global_col = global_col_start + local_col_start;

        bool is_valid_row = local_row < valid_rows_in_tile;
        bool in_bounds = is_valid_row && (global_col < N);

        vec_t_u8_8 input_vec;
        float dequant_scale = 1.0f;

        if (in_bounds) {
            // Load Packed MXFP4
            size_t byte_offset = (size_t)global_row * (N / 2) + (global_col / 2);
            input_vec.cast_load(input + byte_offset); 
            
            // Load Scale (MXFP4 Scale is shared per 32 elements)
            int scale_col_idx = global_col / 32;
            size_t scale_offset = (size_t)global_row * (N / 32) + scale_col_idx;
            uint8_t scale_u8 = __ldg(input_s + scale_offset);
            dequant_scale = decode_mxfp4_scale(scale_u8);
        }

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            int current_local_col = local_col_start + i;
            
            if (!in_bounds || (global_col + i) >= N) {
                // Padding with 0
                smem_dequant[current_local_col][local_row] = 0.0f;
                continue;
            }

            // Unpack Nibbles (Nibble Order Matches Triton)
            const uint8_t packed = input_vec[i / 2];
            const bool is_low = (i % 2 == 0);
            const uint8_t em = is_low ? (packed & 0x07) : ((packed >> 4) & 0x07);
            const bool sign = is_low ? ((packed & 0x08) != 0) : ((packed & 0x80) != 0);

            // Decode E2M1 to Float
            uint16_t x = (static_cast<uint16_t>(em) << (dst_m_bits - 1)) | (sign ? 0x8000 : 0);
            if ((em & 0x06) != 0) x += ((dst_bias - 1) << dst_m_bits);
            if (em == 0x01) x = dst_0p5 | (x & 0x8000);
            
            float val = __bfloat162float(__ushort_as_bfloat16(x));
            
            // Apply MXFP4 Dequant Scale
            smem_dequant[current_local_col][local_row] = val * dequant_scale;      
        }
    }
    
    __syncthreads();

    if (tid < TILE_N) {
        float max_val = 0.0f;
        
        #pragma unroll
        for (int row = 0; row < TILE_M; ++row) {
            float val = fabsf(smem_dequant[tid][row]);
            max_val = fmaxf(max_val, val);
        }


        float scale = compute_te_pow2_scale(max_val, eps);
        
        smem_scale[tid] = scale; //modified 1.0 / 

        int global_col = global_col_start + tid;
        if (global_col < N) {
            int global_scale_row = current_scale_start_row + block_m_in_split;
            output_s[(size_t)global_scale_row * N + global_col] = 1.0f / scale; //1.0f / scale; //
        }
    }
    __syncthreads();

    using vec_t_out = omo::vec_t<DST_DTYPE, VEC_SIZE>;

    int local_col = tid / (TILE_M / VEC_SIZE); // 0..127
    int row_group_idx = tid % (TILE_M / VEC_SIZE); // 0..7
    int local_row_start = row_group_idx * VEC_SIZE; // 0, 16, ..., 112

    if (local_col < TILE_N) {
        int global_col = global_col_start + local_col;
        float quant_scale = smem_scale[local_col]; 

        vec_t_out output_vec;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            int local_row = local_row_start + i;
            float f_val = smem_dequant[local_col][local_row];
            
            // Quantize
            float q_val = f_val * quant_scale;
            
            // [TE Alignment] Explicit Clamping
            q_val = fminf(fmaxf(q_val, FP8_E4M3_MIN), FP8_E4M3_MAX);
            
            output_vec[i] = static_cast<DST_DTYPE>(q_val);
        }

        if (global_col < N) {

            size_t split_base_offset = (size_t)current_split_start_row * N;

            int row_in_split = block_m_in_split * TILE_M + local_row_start;

            size_t output_offset = split_base_offset + 
                                   (size_t)global_col * current_split_len + 
                                   row_in_split;

            bool is_within_split = (local_row_start + VEC_SIZE <= valid_rows_in_tile);
            bool is_aligned = (output_offset % 16 == 0); 

            if (is_within_split && is_aligned) {
                output_vec.cast_store(output_q + output_offset);
            } else {
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                    if (local_row_start + i < valid_rows_in_tile) {
                        output_q[output_offset + i] = output_vec[i];
                    }
                }
            }
        }
    }
}


// ========================================================================
// Host Wrapper
// ========================================================================
namespace omo {

    void omo_per_token_Mxfp4RowToBf16ToFp8Col_msplit(
        torch::Tensor input,          // [Total_M, N/2]
        torch::Tensor input_s,        // [Total_M, N/32]
        torch::Tensor output_q,       // [N, Total_M]
        torch::Tensor output_s,       // [N, Sum_Scale_Blocks]
        torch::Tensor m_splits,       // [Num_Splits] int32
        torch::Tensor split_offsets,  
        torch::Tensor scale_offsets, 
        int max_split_len,            
        float eps) {
    
        // 1. 基础检查
        CHECK_INPUT(input);
        CHECK_INPUT(output_q);
        CHECK_INPUT(m_splits);
        CHECK_INPUT(split_offsets);
        CHECK_INPUT(scale_offsets);
    
        int Total_M = input.size(0);
        int N = input.size(1) * 2;
        int num_splits = m_splits.size(0);
    
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        dim3 grid(num_splits, (max_split_len + 127) / 128, (N + 127) / 128);
        dim3 block(1024);
        

        size_t smem_size = 68096; 
    
        cudaFuncSetAttribute(
            (void*)omo_per_token_Mxfp4RowToBf16ToFp8Col_split_kernel<uint8_t, __nv_fp8_e4m3>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    
        // Launch Kernel
        omo_per_token_Mxfp4RowToBf16ToFp8Col_split_kernel<uint8_t, __nv_fp8_e4m3>
            <<<grid, block, smem_size, stream>>>(
                static_cast<const uint8_t*>(input.data_ptr()),
                static_cast<const uint8_t*>(input_s.data_ptr()),
                static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
                static_cast<float*>(output_s.data_ptr()),
                static_cast<const int*>(m_splits.data_ptr()),
                static_cast<const int*>(split_offsets.data_ptr()), 
                static_cast<const int*>(scale_offsets.data_ptr()), 
                N, 
                Total_M, 
                eps
            );
    }
    
    } // namespace omo