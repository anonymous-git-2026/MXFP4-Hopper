#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include "utils.h"
#include "vec_dtypes.cuh"

constexpr uint16_t dst_bias = 127;
constexpr uint16_t dst_0p5 = 16128;      // 0.5 in bfloat16 bits
constexpr uint16_t dst_m_bits = 7;
const float FP8_MAX = 448.0f;
const float FP8_MIN = -FP8_MAX;

// Helper: MXFP4 ue8m0 -> bfloat16
__device__ __forceinline__ __nv_bfloat16 ue8m0_to_bfloat16_direct(uint8_t ue8m0_value) {
    uint16_t bf16_bits = static_cast<uint16_t>(ue8m0_value) << 7;
    return __ushort_as_bfloat16(bf16_bits);
}


// Helper: to_float overload
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Atomic max for float (renamed to avoid conflict)
__device__ void myAtomicMaxFloat(float* address, float val) {
    int* addr = reinterpret_cast<int*>(address);
    int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// //////////////////////////////////////////////////////////////////////////
template <
    typename T,
    typename DST_DTYPE = __nv_fp8_e4m3>
__global__ void omo_per_token_Mxfp4RowToBf16ToFp8Col_kernel(
    const uint8_t* __restrict__ input,          // [M, N/2]
    const uint8_t* __restrict__ input_s,        // [M, N/32]
    DST_DTYPE* __restrict__ output_q,           // [N, M]  <-- transposed!
    float* __restrict__ output_s,               // [N, M/128]
    const int M,
    const int N,
    const float eps) {

    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int THREADS_PER_BLOCK = 1024;
    constexpr int ELEMENTS_PER_THREAD = (TILE_M * TILE_N) / THREADS_PER_BLOCK; // = 16
    constexpr int MXFP4_ELEMENTS_PER_LOAD = 8; // 8 bytes = 16 elements

    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float (*smem_dequant)[TILE_N + 1] = (float (*)[TILE_N + 1])smem;
    float* smem_absmax = smem + TILE_M * (TILE_N + 1);

    const int global_row_start = block_row * TILE_M;
    const int global_col_start = block_col * TILE_N;
    using vec_t_u8_8 = omo::vec_t<uint8_t, 8>;
    

    int linear_idx_start = tid * ELEMENTS_PER_THREAD;
    int local_row = linear_idx_start / TILE_N;
    int local_col_start = linear_idx_start % TILE_N;

    
    if (local_row < TILE_M && local_col_start < TILE_N) {
        int global_row = global_row_start + local_row;
        int global_col = global_col_start + local_col_start;

        vec_t_u8_8 input_vec;
        float dequant_scale = 1.0f;
        
        bool in_bounds = (global_row < M) && (global_col < N);
        
        if (in_bounds) {

            size_t byte_offset = (size_t)global_row * (N / 2) + (global_col / 2);
            

            input_vec.cast_load(input + byte_offset); 

            int scale_col_idx = global_col / 32;
            size_t scale_offset = (size_t)global_row * (N / 32) + scale_col_idx;
            uint8_t scale_u8 = __ldg(input_s + scale_offset);
            dequant_scale = to_float(ue8m0_to_bfloat16_direct(scale_u8));
        } 


        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            int current_local_col = local_col_start + i;
            
            if (global_row >= M || (global_col + i) >= N) {
                smem_dequant[current_local_col][local_row] = 0.0f;
                continue;
            }


            const uint8_t packed = input_vec[i / 2];

            const bool is_low = (i % 2 == 0); 
            const uint8_t em = is_low ? (packed & 0x07) : ((packed >> 4) & 0x07);
            const bool sign = is_low ? ((packed & 0x08) != 0) : ((packed & 0x80) != 0);

            uint16_t x = (static_cast<uint16_t>(em) << (dst_m_bits - 1)) | (sign ? 0x8000 : 0);
            
            if ((em & 0x06) != 0) {
                x += ((dst_bias - 1) << dst_m_bits);
            }
            if (em == 0x01) {
                x = dst_0p5 | (x & 0x8000);
            }
            
            __nv_bfloat16 bf16_val = __ushort_as_bfloat16(x);

            float dequant_val = to_float(bf16_val) * dequant_scale;
            smem_dequant[current_local_col][local_row] = dequant_val;        
        }
    }


    if (tid < TILE_N) {
        float max_val = eps;
        #pragma unroll
        for (int row = 0; row < TILE_M; ++row) {
            float val = fabsf(smem_dequant[tid][row]);
            max_val = fmaxf(max_val, val);
        }
        smem_absmax[tid] = max_val;

        int global_col = global_col_start + tid;
        if (global_col < N) {
            output_s[(size_t)block_row * N + global_col] = max_val / FP8_MAX;
        }
    }
    __syncthreads();

    constexpr int VEC_SIZE = 16; 
    using vec_t_out = omo::vec_t<DST_DTYPE, VEC_SIZE>;

    int local_col = tid / (TILE_M / VEC_SIZE); // tid / 8

    int row_group_idx = tid % (TILE_M / VEC_SIZE); // tid % 8
    int local_row_start = row_group_idx * VEC_SIZE;

    if (local_col < TILE_N) {
        int global_col = global_col_start + local_col;
        int global_row_base = global_row_start + local_row_start;

        float max_val = smem_absmax[local_col];

        float quant_scale = (max_val > 1e-9f) ? (FP8_MAX / max_val) : 0.0f;

        vec_t_out output_vec;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            int local_row = local_row_start + i;


            float f_val = smem_dequant[local_col][local_row];


            float q_val = f_val * quant_scale;
            q_val = fminf(fmaxf(q_val, FP8_MIN), FP8_MAX);


            output_vec[i] = static_cast<DST_DTYPE>(q_val);
        }


        if (global_col < N && (global_row_base + VEC_SIZE) <= M) {

            size_t output_offset = (size_t)global_col * M + global_row_base;

            output_vec.cast_store(output_q + output_offset);
        } 
        else if (global_col < N) {

            for (int i = 0; i < VEC_SIZE; ++i) {
                int r = global_row_base + i;
                if (r < M) {
                    output_q[global_col * M + r] = output_vec[i];
                }
            }
        }
    }
}





namespace omo {
  void omo_per_token_Mxfp4RowToBf16ToFp8Col(
      torch::Tensor input,      // [M, N/2], uint8
      torch::Tensor input_s,    // [M, N/32], uint8
      torch::Tensor output_q,   // [N, M], fp8
      torch::Tensor output_s,   // [N, M/128], float
      float eps) {
  
      CHECK_INPUT(input);
      CHECK_INPUT(output_q);
      CHECK_EQ(input.dim(), 2);
      CHECK_EQ(output_q.dim(), 2);
  
      int M = input.size(0);
      int N = input.size(1) * 2; // recover original N
  
      CHECK_EQ(input_s.size(0), M);
      CHECK_EQ(input_s.size(1), (N + 31) / 32);
      CHECK_EQ(output_q.size(0), N);
      CHECK_EQ(output_q.size(1), M);
      CHECK_EQ(output_s.size(0), (M + 127) / 128);// ceil(M/128)
      CHECK_EQ(output_s.size(1), N); 
  


      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
      dim3 grid((M + 127) / 128, (N + 127) / 128);
      dim3 block(1024);
  
      cudaFuncSetAttribute(
        (void*)omo_per_token_Mxfp4RowToBf16ToFp8Col_kernel<uint8_t, __nv_fp8_e4m3>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        68096//82944//66560 //  //66048
    );
      omo_per_token_Mxfp4RowToBf16ToFp8Col_kernel<uint8_t, __nv_fp8_e4m3>
          <<<grid, block, 68096, stream>>>(
              static_cast<const uint8_t*>(input.data_ptr()),
              static_cast<const uint8_t*>(input_s.data_ptr()),
              static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
              static_cast<float*>(output_s.data_ptr()),
              M, N, eps
          );
  }
  }