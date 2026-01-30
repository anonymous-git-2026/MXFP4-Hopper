#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include "utils.h"
#include "vec_dtypes.cuh"
constexpr uint32_t E8_BIAS = 127;
constexpr uint32_t E2_BIAS = 1;
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

  #define COMPUTE_E2M1_DATA(exponents, m2bits, round_inc) \
  ((((exponents << 2) | m2bits) + round_inc) >> 1)

__device__ __forceinline__ uint8_t convert_fp32_to_e2m1_fast(float val, float y_s, 
  float min_4bit, float max_4bit) {
float q_val = fminf(fmaxf(val / y_s, min_4bit), max_4bit);

uint32_t bits = __float_as_uint(q_val);

uint32_t sign = bits >> 28;  
uint32_t exp = (bits >> 23) & 0xFF;
uint32_t mant = bits & 0x7FFFFF;

const uint32_t bias_diff = 127 - 1;  // E8_BIAS - E2_BIAS
uint32_t is_denorm = (exp == 0) & (mant != 0);
mant = (mant | (is_denorm << 23));  
exp = (exp - is_denorm) - bias_diff * (exp >= bias_diff);

uint32_t m2 = mant >> 21;
uint32_t round = ((m2 & 1) & ((mant & 0x1FFFFF) != 0 | (m2 >> 1))) & 1;

uint32_t result = ((exp << 2) | m2) + round;
return static_cast<uint8_t>(sign | min(result >> 1, 7u));
}

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {


  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 2));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 1));
  return val;
}

__device__ __forceinline__ float warpSegmentReduceMaxPrecise(float value, int start_lane, int num_threads) {
  unsigned mask = ((1u << num_threads) - 1) << start_lane;
  int lane_id = threadIdx.x % 32;
  int relative_lane = lane_id - start_lane;
  
  if (relative_lane >= 0 && relative_lane < num_threads) {
      for (int offset = num_threads / 2; offset > 0; offset >>= 1) {
          if (relative_lane < offset) {
              float other = __shfl_sync(mask, value, lane_id + offset);
              value = fmaxf(value, other);
          }
          __syncwarp(mask);
      }
      
      value = __shfl_sync(mask, value, start_lane);
  }
  
  return value;
}

__device__ __forceinline__ float warpReduceMax_in(float value) {
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 16));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 8));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 4));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 2));
  value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 1));
  return value;
}
__device__ __forceinline__ void adjust_denormal(uint32_t& mantissa, uint32_t exponent)
{
    uint32_t shift = (E8_BIAS - (exponent + 1));   // adjusted_exponents
    if (exponent < E8_BIAS)              
    {
        mantissa = ((0x400000u | (mantissa >> 1)) >> shift);

    }
    else{
      mantissa = mantissa;
    }
}
__device__ __forceinline__ void adjust_denormal_opt2(uint32_t& mantissa, uint32_t exponent) {
  uint32_t shift = E8_BIAS - (exponent + 1);
  
  uint32_t denormal_mantissa = (0x400000u | (mantissa >> 1)) >> shift;
  

  uint32_t mask = -(exponent < E8_BIAS); 
  
  mantissa = (denormal_mantissa & mask) | (mantissa & ~mask);
}


template <
    typename T,
    typename DST_DTYPE = uint8_t,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_4bit_kernel_coalescing(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    void* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_4bit,
    const float max_4bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {


  constexpr uint32_t vec_size = 16 / sizeof(nv_bfloat16);

  using vec_t = omo::vec_t<nv_bfloat16, vec_size>;

  const int threads_per_group = group_size / vec_size;

  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;
  const int mxfp4_lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;
  const int64_t uint8_block_group_offset = global_group_id * group_size / 2;
  // const int64_t uint8_block_group_scale_offset = global_group_id * group_size / 32;

  float local_absmax = eps;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  // static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);


  const T* group_input = input + block_group_offset;
  // DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + uint8_block_group_offset;

  DST_DTYPE* scale_output = static_cast<DST_DTYPE*>(output_s) + global_group_id;

  

  vec_t input_vec;
  // input_vec.cast_load(group_input + mxfp4_lane_id * vec_size);
  input_vec.load_global_acquire((nv_bfloat16*)(group_input) + mxfp4_lane_id * vec_size);


  #pragma unroll
  for (uint32_t j = 0; j < vec_size; ++j) {
    float val = static_cast<float>(to_float(input_vec[j]));
    float abs_val = fabsf(val);
    local_absmax = fmaxf(local_absmax, abs_val);
  }


  local_absmax = GroupReduceMax(local_absmax,lane_id);


  static_assert(sizeof(float) == 4, "float is 32 bits");



  float y_s = local_absmax / max_4bit;

  y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
  uint8_t y_s_quant;



  y_s_quant = (uint8_t)(((int)log2f(y_s)) + 127);

  if (mxfp4_lane_id == 0) {
    *scale_output = y_s_quant;
  }

  
    // opt
    const float inv_y_s = 1.0f / y_s;
    const uint32_t lo = E8_BIAS - E2_BIAS;
    uint32_t packed_output = 0;  
    #pragma unroll
    for (uint32_t j = 0; j < vec_size; j = j + 2) {
        float2 input_vals;

        input_vals.x = static_cast<float>(to_float(input_vec[j]));
        input_vals.y = static_cast<float>(to_float(input_vec[j+1]));

        float q_val_x = fminf(fmaxf(input_vals.x * inv_y_s, min_4bit), max_4bit);
        float q_val_y = fminf(fmaxf(input_vals.y * inv_y_s, min_4bit), max_4bit);

        uint32_t float_bits_x = __float_as_int(q_val_x);
        uint32_t float_bits_y = __float_as_int(q_val_y);
        


        uint32_t signs_x = float_bits_x & 0x80000000;
        uint32_t signs_y = float_bits_y & 0x80000000;
        uint32_t exponents_x = (float_bits_x >> 23) & 0xFF;
        uint32_t exponents_y = (float_bits_y >> 23) & 0xFF;
        uint32_t mantissas_x = (float_bits_x & 0x7FFFFF);
        uint32_t mantissas_y = (float_bits_y & 0x7FFFFF);


        adjust_denormal_opt2(mantissas_x, exponents_x);
        adjust_denormal_opt2(mantissas_y, exponents_y);

        exponents_x = std::max(exponents_x, lo) - lo;
        exponents_y = std::max(exponents_y, lo) - lo;

        uint32_t m2bits_x     = mantissas_x >> 21;      
        uint32_t m2bits_y     = mantissas_y >> 21;      
        uint32_t lsb_keep_x = (m2bits_x >> 1) & 1;
        uint32_t lsb_keep_y = (m2bits_y >> 1) & 1;
        uint32_t guard_x = m2bits_x & 1;
        uint32_t guard_y = m2bits_y & 1;
        uint32_t sticky_x = (mantissas_x & 0x1FFFFF) != 0;
        uint32_t sticky_y = (mantissas_y & 0x1FFFFF) != 0;
        uint32_t round_inc_x = guard_x & (sticky_x | lsb_keep_x);
        uint32_t round_inc_y = guard_y & (sticky_y | lsb_keep_y);


        uint32_t result_x = COMPUTE_E2M1_DATA(exponents_x, m2bits_x, round_inc_x);
        uint32_t e2m1_data_x = (result_x > 0x7u) ? 0x7u : result_x;

        uint8_t e2m1_value_x = static_cast<uint8_t>(
            (signs_x >> 28) | e2m1_data_x);
            
        uint32_t result_y = COMPUTE_E2M1_DATA(exponents_y, m2bits_y, round_inc_y);
        uint32_t e2m1_data_y = (result_y > 0x7u) ? 0x7u : result_y;

        uint8_t e2m1_value_y = static_cast<uint8_t>(
            (signs_y >> 28) | e2m1_data_y);

        uint8_t e2m1_value = (e2m1_value_x & 0x0F) | ((e2m1_value_y & 0x0F) << 4);
        

        packed_output |= (static_cast<uint32_t>(e2m1_value) << ((j / 2) * 8));
    }
    uint32_t* group_output_u32 = reinterpret_cast<uint32_t*>(group_output);
    group_output_u32[mxfp4_lane_id] = packed_output;  



}


template <
    typename T,
    typename DST_DTYPE = uint8_t,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_4bit_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    void* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_4bit,
    const float max_4bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 32;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;
  // const int mxfp4_lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;
  const int64_t uint8_block_group_offset = global_group_id * group_size / 2;
  // const int64_t uint8_block_group_scale_offset = global_group_id * group_size / 32;

  float local_absmax = eps;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);



  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + uint8_block_group_offset;

  DST_DTYPE* scale_output = static_cast<DST_DTYPE*>(output_s) + global_group_id;


  float in;
  in =  static_cast<float>(to_float(*(group_input + lane_id)));

  static_assert(sizeof(float) == 4, "float is 32 bits");
  float in_max = fabsf(in);
  local_absmax = warpReduceMax_in(in_max);

  


  float y_s = local_absmax / max_4bit;
  // if constexpr (SCALE_UE8M0) {
  //   y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
  // }
  y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
  scale_element_t y_s_quant;
  y_s_quant = (uint8_t)(((int)log2f(y_s)) + 127);

  if (lane_id == 0) {
    *scale_output = y_s_quant;
      // asm volatile("st.global.cg.b8 [%0], %1;" :: "l"(scale_output), "r"((uint8_t)y_s_quant));
    // __stcg(scale_output, y_s_quant);
  }


  float val = static_cast<float>(to_float(in));
  // if(blockIdx.x == 0){
  //   printf("pid = %d, val = %f\n", threadIdx.x, val);
  // }
  float q_val;
  if( val == 0){
    q_val = 0;
  }
  else{
    q_val = fminf(fmaxf(val / y_s, min_4bit), max_4bit);
  }
  // uint8_t out = 0;
  uint32_t float_bits = *(reinterpret_cast<uint32_t*>(&q_val));

  uint32_t signs = float_bits & 0x80000000;
  uint32_t exponents = (float_bits >> 23) & 0xFF;
  uint32_t mantissas = (float_bits & 0x7FFFFF);


  adjust_denormal(mantissas, exponents);

  const uint32_t lo = E8_BIAS - E2_BIAS;
  exponents = std::max(exponents, lo) - lo;

  uint32_t m2bits     = mantissas >> 21;      
  uint32_t lsb_keep = (m2bits >> 1) & 1;
  uint32_t guard = m2bits & 1;
  uint32_t sticky = (mantissas & 0x1FFFFF) != 0;
  uint32_t round_inc = guard & (sticky | lsb_keep);

  uint8_t e2m1_value = static_cast<uint8_t>(
    ((signs >> 28)) |
    (std::min((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7u)));


  uint8_t shuffled_value = __shfl_sync(0xFFFFFFFF, e2m1_value, lane_id ^ 1);
  if(lane_id % 2 == 0){

      uint8_t odd_out = shuffled_value;
      e2m1_value = (e2m1_value & 0x0F)  | ((odd_out & 0x0F) << 4);
  }

  if (lane_id % 2 == 0) {  
    int output_index = lane_id >> 1;  
    group_output[output_index] = e2m1_value;
  }


}

namespace omo{
void omo_per_token_group_quant_fp4(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int group_size,
    float eps,
    float min_4bit,
    float max_4bit,
    bool scale_ue8m0 = false) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);



  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int groups_per_block = 1;


  groups_per_block = calculateGroupsConstexpr(num_groups);

  auto dst_type = output_q.scalar_type();
  // printf("dst_type = %s\n", toString(dst_type));
  const int opt = 1;
  constexpr int THREADS_PER_GROUP = (opt == 1) ? 4 : 32;
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;





  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int hidden_dim = input.size(input.dim() - 1);
  const int num_groups_per_row = hidden_dim / group_size;
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                               \
  do {                                                                                            \
    dim3 grid(num_blocks);                                                                        \
    dim3 block(num_threads);                                                                      \
    if (is_column_major) {                                                                        \
      if (scale_ue8m0) {                                                                          \
        per_token_group_quant_4bit_kernel<T, DST_DTYPE, true, true><<<grid, block, 0, stream>>>(  \
            static_cast<T*>(input.data_ptr()),                                                    \
            output_q.data_ptr(),                                                                  \
            static_cast<uint32_t*>(output_s.data_ptr()),                                          \
            group_size,                                                                           \
            num_groups,                                                                           \
            groups_per_block,                                                                     \
            (float)eps,                                                                           \
            (float)min_4bit,                                                                      \
            (float)max_4bit,                                                                      \
            num_groups_per_row,                                                                   \
            scale_stride);                                                                        \
      } else {                                                                                    \
        per_token_group_quant_4bit_kernel<T, DST_DTYPE, true, false><<<grid, block, 0, stream>>>( \
            static_cast<T*>(input.data_ptr()),                                                    \
            output_q.data_ptr(),                                                                  \
            static_cast<float*>(output_s.data_ptr()),                                             \
            group_size,                                                                           \
            num_groups,                                                                           \
            groups_per_block,                                                                     \
            (float)eps,                                                                           \
            (float)min_4bit,                                                                      \
            (float)max_4bit,                                                                      \
            num_groups_per_row,                                                                   \
            scale_stride);                                                                        \
      }                                                                                           \
    } else {                                                                                      \
      assert(!scale_ue8m0);                                                                       \
      per_token_group_quant_4bit_kernel_coalescing<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(         \
          static_cast<T*>(input.data_ptr()),                                                      \
          output_q.data_ptr(),                                                                    \
          static_cast<uint8_t *>(output_s.data_ptr()),                                               \
          group_size,                                                                             \
          num_groups,                                                                             \
          groups_per_block,                                                                       \
          (float)eps,                                                                             \
          (float)min_4bit,                                                                        \
          (float)max_4bit);                                                                       \
    }                                                                                             \
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::Byte) {
      LAUNCH_KERNEL(scalar_t, uint8_t);
      return true;
    } 

    return false;
  });


#undef LAUNCH_KERNEL
}

}