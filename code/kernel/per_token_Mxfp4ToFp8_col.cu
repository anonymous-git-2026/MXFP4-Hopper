
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include "utils.h"
#include "vec_dtypes.cuh"
constexpr uint32_t E8_BIAS = 127;
constexpr uint32_t E2_BIAS = 1;
// BF16 constants
constexpr uint16_t BF16_EXP_BIAS = 127;
constexpr uint16_t MBITS_BF16 = 7;
constexpr uint16_t EBITS_BF16 = 8;

const uint16_t dst_bias = 127;
const uint16_t dst_0p5 = 16128;
const uint16_t dst_m_bits = 7;
const float FP8_MAX = 448;
const float FP8_MIN = -FP8_MAX;
#define DIV_CEIL(n, d) (((n) + (d) - 1) / (d))

__device__ __inline__ __nv_bfloat16 ue8m0_to_bfloat16_direct(uint8_t ue8m0_value) {
  uint16_t biased_exp = static_cast<uint16_t>(ue8m0_value);
  
  uint16_t bf16_bits = (biased_exp << 7);
  return __ushort_as_bfloat16(bf16_bits);
}

__device__ __inline__  float ue8m0_to_float_direct(uint8_t uint8_value) {

  uint32_t biased_exp = static_cast<uint32_t>(uint8_value);
  
  uint32_t float_bits = (biased_exp << 23);
  return __uint_as_float(float_bits);
}


__device__ __inline__  float scale_to_float(uint8_t scale) {
  uint32_t scale_uint32 = static_cast<uint32_t>(scale) << 23;
  float scale_output;
  std::memcpy(&scale_output, &scale_uint32, sizeof(float));
  return scale_output;
}

__device__ __inline__ float ue8m0_to_float_direct_v2(uint8_t uint8_value) {

  uint32_t float_bits = (static_cast<uint32_t>(uint8_value) << 23);
  return __uint_as_float(float_bits) * 0x1p-127f;  
}

__device__ __inline__ float ue8m0_to_float(uint8_t val) {
  return exp2f(static_cast<float>(val) - 127.0f);
}

__device__ __inline__  float uint16_to_float_direct(uint16_t uint16_value) {

  
  uint32_t sign = static_cast<uint32_t>((uint16_value >> 15) & 0x1);

  uint32_t biased_exp = static_cast<uint32_t>((uint16_value >> 7) & 0xFF);
  

  uint32_t mantissa = static_cast<uint32_t>(uint16_value & 0x7F) << 16;

  uint32_t float_bits = (sign << 31) | (biased_exp << 23) | mantissa;
  return __uint_as_float(float_bits);
}

constexpr int calculateGroupsConstexpr(int num_groups) {
  if (num_groups % 16 == 0) return 16;
  if (num_groups % 8 == 0) return 8;
  if (num_groups % 4 == 0) return 4;
  if (num_groups % 2 == 0) return 2;
  return 1;
}
__device__ __forceinline__ float to_float(__half x) {
    return __half2float(x);
  }
  
  __device__ __forceinline__ float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
  
  template <typename T>
  __device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
  }



__device__ __forceinline__ uint8_t GroupReduceMax(uint8_t val) {

  val = max(val, __shfl_xor_sync(FULL_MASK, val, 16));
  val = max(val, __shfl_xor_sync(FULL_MASK, val, 8));
  val = max(val, __shfl_xor_sync(FULL_MASK, val, 4));
  val = max(val, __shfl_xor_sync(FULL_MASK, val, 2));
  val = max(val, __shfl_xor_sync(FULL_MASK, val, 1));
  return val;
}


template <
    typename T,
    typename DST_DTYPE = __nv_fp8_e4m3,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void omo_per_token_Mxfp4ToFp8_col_kernel(
    const T* __restrict__ input,
    const T* __restrict__ input_s,
    void* __restrict__ output_q,
    void* __restrict__ output_s,
    const int M,
    const int K,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_4bit,
    const float max_4bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {

  constexpr uint32_t vec_size_in = 4 / sizeof(T);
  constexpr uint32_t vec_size_scale_in = 1 / sizeof(T);
  constexpr uint32_t vec_size_out = 16 / sizeof(T);
  constexpr uint32_t vec_size_scale = 128 / 32  / sizeof(T);

  using vec_t = omo::vec_t<T, vec_size_in>;
  using vec_t_scale = omo::vec_t<T, vec_size_scale>;


  const int TILE_DIM = 128;
  __shared__ uint8_t s_mem[TILE_DIM][TILE_DIM + 1];
  __shared__ uint8_t s_mem_scale[TILE_DIM][TILE_DIM * 2  / 32 + 1];
  __shared__ uint8_t s_mem_scale_adjust[TILE_DIM][TILE_DIM * 2  / 32 + 1];


  uint8_t local_absmax = eps;

  uint8_t warp_absmax = eps;


  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;


  vec_t input_vec;
  vec_t_scale scale_vec;
  // vec_t_scale scale_vec_adjust;
  // wrap num = 1024/32 = 32
  const int row_per_warp = 128 / 32;
  const int warp_id = threadIdx.x / 32;
  const int land_id = threadIdx.x % 32;

  __syncthreads();

  // load input from global memory
  #pragma unroll
  for(int i = 0; i < row_per_warp; i++){
    #pragma unroll
    for(int j = 0; j < vec_size_in; j++)
    {
      s_mem[i + warp_id * row_per_warp][land_id * vec_size_in + j ] = 
        static_cast<const uint8_t*>(input)[blockIdx.y * K * TILE_DIM + blockIdx.x * TILE_DIM + (i + warp_id * row_per_warp) * K +  land_id * vec_size_in + j];
    }
    
  }
  __syncthreads();

  for(int i = 0; i < row_per_warp; i++){
    #pragma unroll
    for(int j = 0; j < vec_size_scale_in; j++)
    {
      s_mem_scale[i + warp_id * row_per_warp][land_id / 4  * vec_size_scale_in + j ] = 
        static_cast<const uint8_t*>(input_s)[blockIdx.y * K * 2 / 32 * TILE_DIM + blockIdx.x * TILE_DIM * 2  / 32  + (i + warp_id * row_per_warp) * K * 2 / 32 +  land_id  / 4 * vec_size_scale_in + j];
    }
    
  }
  __syncthreads();

    #pragma unroll
    for(int i = 0; i < vec_size_scale; i++){
      // uint8_t val = s_mem_scale[warp_id + (threadIdx.x * 4  + i) * (TILE_DIM / 32 + 1)];
      uint8_t val = s_mem_scale[land_id * 4 + i][warp_id / 4];
      uint8_t abs_val = abs(val);
      local_absmax = max(local_absmax, abs_val);
    }
  __syncthreads();



  // warp_absmax = GroupReduceMax(local_absmax);
  warp_absmax = GroupReduceMax(local_absmax) - 6;
  // uint8_t scale_adjust = warp_absmax - local_absmax_uint8;
  __syncthreads();
  

  

  float y_s = scale_to_float(warp_absmax);

  __syncthreads();
  
  


  // for TE scale format: row-major
  if(warp_id % 4 == 0){
    static_cast<float*>(output_s)[blockIdx.y * (K) * 2  + blockIdx.x * TILE_DIM * 2 +  land_id + warp_id / 4 * 32] = y_s;
  }

  __syncthreads();
  if(warp_id % 4 == 0){
    #pragma unroll
    for(int i = 0; i < vec_size_scale; i++){
      s_mem_scale_adjust[land_id * 4 + i][warp_id / 4] = warp_absmax - s_mem_scale[land_id * 4 + i][warp_id / 4] ;
    }
  }
  


  __syncthreads();




#pragma unroll
for(int i = 0; i < row_per_warp; i++){ 
  #pragma unroll
  for (uint32_t j = 0; j < vec_size_in; j = j + 1) {

      int smem_row_idx = land_id * vec_size_in + j; 

      int smem_col_idx = warp_id * row_per_warp + i;

      uint8_t val = static_cast<uint8_t>(s_mem[smem_row_idx][smem_col_idx]);


      uint8_t scale = s_mem_scale_adjust[smem_row_idx][smem_col_idx / 16]; 


      uint8_t em0 = val & 0x07;
      uint8_t em1 = val & 0x70;
      
      uint16_t x0 = (static_cast<uint16_t>(em0) << (dst_m_bits - 1)) | 
                    ((val & 0x08) << 12);
      uint16_t x1 = (static_cast<uint16_t>(em1) << (dst_m_bits - 5)) | 
                    ((val & 0x80) << 8);


      if ((em0 & 0x06) != 0) {
          x0 += ((dst_bias - 1) << dst_m_bits);
      }
      if ((em1 & 0x60) != 0) {
          x1 += ((dst_bias - 1) << dst_m_bits);
      }


      if (em0 == 0x01) {
          x0 = dst_0p5 | (x0 & 0x8000); 
      }
      if (em1 == 0x10) {
          x1 = dst_0p5 | (x1 & 0x8000); 
      }



      uint16_t adjusted_bfloat16_bits_0 = x0 - (static_cast<uint16_t>(scale) << dst_m_bits);
      uint16_t adjusted_bfloat16_bits_1 = x1 - (static_cast<uint16_t>(scale) << dst_m_bits);

      uint16_t positive_bfloat16_bits_0 = adjusted_bfloat16_bits_0 & 0x7FFF;
      uint16_t positive_bfloat16_bits_1 = adjusted_bfloat16_bits_1 & 0x7FFF;

      float q_val_0 = to_float(__ushort_as_bfloat16(positive_bfloat16_bits_0));
      float q_val_1 = to_float(__ushort_as_bfloat16(positive_bfloat16_bits_1));

      q_val_0 = fminf(fmaxf(q_val_0, -448.0f), 448.0f);
      q_val_1 = fminf(fmaxf(q_val_1, -448.0f), 448.0f);

      int target_col = blockIdx.y * TILE_DIM + smem_row_idx;

      int target_row_base = blockIdx.x * (TILE_DIM * 2) + (smem_col_idx * 2);

      size_t addr_0 = (size_t)target_row_base * M + target_col;
      size_t addr_1 = (size_t)(target_row_base + 1) * M + target_col;

      static_cast<__nv_fp8_e4m3*>(output_q)[addr_0] = (__nv_fp8_e4m3)(q_val_0);
      static_cast<__nv_fp8_e4m3*>(output_q)[addr_1] = (__nv_fp8_e4m3)(q_val_1);
  }
}

// __syncthreads();






}


namespace omo{
void omo_per_token_Mxfp4ToFp8_col(
    torch::Tensor input,
    torch::Tensor input_s,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int group_size,
    float eps,
    float min_4bit,
    float max_4bit,
    bool scale_ue8m0 = false) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);


  group_size = 128;
  const int num_groups = input.numel() * 2 / group_size;
  const int M = input.size(0);
  const int K = input.size(1);

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // constexpr int groups_per_block = 16;
  int groups_per_block = 1;

  groups_per_block = calculateGroupsConstexpr(num_groups);

  auto dst_type = output_q.scalar_type();

  const int input_y = input.size(0);
  const int input_x = input.size(1);



  const int block_y = DIV_CEIL(input_y , group_size);
  const int block_x = DIV_CEIL(input_x , group_size);
  const int num_threads = 1024;
  const dim3 block(num_threads);
  const dim3 grid(block_x,block_y);







  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int hidden_dim = input.size(input.dim() - 1);
  const int num_groups_per_row = hidden_dim / group_size;
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                               \
  do {                                                                                            \
      assert(!scale_ue8m0);                                                                       \
      omo_per_token_Mxfp4ToFp8_col_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(     \
          static_cast<uint8_t *>(input.data_ptr()),                                               \
          static_cast<uint8_t *>(input_s.data_ptr()),                                               \
          static_cast<__nv_fp8_e4m3 *>(output_q.data_ptr()),                                      \
          static_cast<float *>(output_s.data_ptr()),                                              \
          M,                                                                                      \
          K,                                                                                      \
          group_size,                                                                             \
          num_groups,                                                                             \
          groups_per_block,                                                                       \
          (float)eps,                                                                             \
          (float)min_4bit,                                                                        \
          (float)max_4bit);                                                                       \         
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_UINT8(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
      return true;
    } 
  });

#undef LAUNCH_KERNEL
}

}