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

__device__ __forceinline__ __nv_bfloat16 ue8m0_to_bfloat16_direct(uint8_t ue8m0_value) {

  uint16_t biased_exp = static_cast<uint16_t>(ue8m0_value);
  

  uint16_t bf16_bits = (biased_exp << 7);
  return __ushort_as_bfloat16(bf16_bits);
}

constexpr int calculateGroupsConstexpr(int num_groups) {
  // if (num_groups % 32 == 0) return 32;
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



__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {

  // val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 8));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 4));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 2));
  val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, 1));
  return val;
}

template <
    typename T,
    typename DST_DTYPE = __nv_fp8_e4m3,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void omo_per_token_Mxfp4ToBf16_dequant_kernel(
    const T* __restrict__ input,
    const T* __restrict__ input_s,
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

  constexpr uint32_t vec_size_in = 8 / sizeof(T); 
  constexpr uint32_t vec_size_scale_in = 8 / sizeof(T); 
  constexpr uint32_t vec_size_out = 16 / sizeof(T); 

  const uint32_t group_num_per_block = 8;  
  __shared__ uint8_t scale_in[group_num_per_block * 16  / 2]; 

  using vec_t = omo::vec_t<T, vec_size_in>;
  using vec_t_out = omo::vec_t<nv_bfloat16, vec_size_out>;

  const int threads_per_group_in = group_size / vec_size_in / 2; 
  const int threads_per_group_out = group_size / vec_size_out; 

  const int64_t local_group_id_in = threadIdx.x / threads_per_group_in; 
  const int64_t local_group_id_out = threadIdx.x / threads_per_group_out;

  const int lane_id = threadIdx.x % threads_per_group_out; 

  const int mxfp4_lane_id = threadIdx.x % threads_per_group_in; 
  const int mxfp4_lane_id_out = threadIdx.x % threads_per_group_out;

  const int64_t block_group_id = blockIdx.x * groups_per_block; 
  const int64_t global_group_id_in = block_group_id + local_group_id_in; 
  const int64_t global_group_id_out = block_group_id + local_group_id_out;
  const int64_t block_group_offset_in = global_group_id_in * group_size / 2; 
  const int64_t block_group_offset_out = global_group_id_out * group_size; 


  float local_absmax_0 = eps;
  float local_absmax_1 = eps;
  float local_absmax = eps;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;


  const T* group_input = input + block_group_offset_in; 
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset_out;

  float* scale_output = static_cast<float*>(output_s) + global_group_id_in;

  

  vec_t input_vec; 
  vec_t_out out_vec; 
  input_vec.cast_load(group_input + mxfp4_lane_id * vec_size_in); 
  scale_in[threadIdx.x / 2] = input_s[blockIdx.x  * blockDim.x / 2 + threadIdx.x / 2];

    __nv_bfloat16 scale_bf16 = ue8m0_to_bfloat16_direct(scale_in[threadIdx.x / 2]);
    float scale_val = to_float(scale_bf16);
    const uint16_t bias_offset = (dst_bias - 1) << dst_m_bits;

    #pragma unroll
    for (uint32_t j = 0; j < vec_size_in; j++) {
        uint8_t val = input_vec[j];
        uint8_t em0 = val & 0x07;
        uint8_t em1 = val & 0x70;
        
        uint16_t x0 = (static_cast<uint16_t>(em0) << (dst_m_bits - 1)) | ((val & 0x08) << 12);
        uint16_t x1 = (static_cast<uint16_t>(em1) << (dst_m_bits - 5)) | ((val & 0x80) << 8);
        
        x0 += (-((em0 & 0x06) != 0) & bias_offset);
        x1 += (-((em1 & 0x60) != 0) & bias_offset);
        
        uint16_t mask0 = -(em0 == 0x01);
        uint16_t mask1 = -(em1 == 0x10);
        x0 = (mask0 & (dst_0p5 | (x0 & 0x8000))) | (~mask0 & x0);
        x1 = (mask1 & (dst_0p5 | (x1 & 0x8000))) | (~mask1 & x1);
        
        *reinterpret_cast<__nv_bfloat162*>(&group_output[mxfp4_lane_id_out * vec_size_out + j * 2]) = 
            __floats2bfloat162_rn(
                __bfloat162float(__ushort_as_bfloat16(x0)) * scale_val,
                __bfloat162float(__ushort_as_bfloat16(x1)) * scale_val
            );
    }


}


namespace omo{
void omo_per_token_Mxfp4ToBf16_dequant(
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

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // constexpr int groups_per_block = 16;
  int groups_per_block = 1;

  groups_per_block = calculateGroupsConstexpr(num_groups);

  auto dst_type = output_q.scalar_type();
  // printf("dst_type = %s\n", toString(dst_type));
  const int opt = 1;
  constexpr int THREADS_PER_GROUP = (opt == 1) ? 8 : 32;
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
      assert(!scale_ue8m0);                                                                       \
      omo_per_token_Mxfp4ToBf16_dequant_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(     \
          static_cast<uint8_t *>(input.data_ptr()),                                               \
          static_cast<uint8_t *>(input_s.data_ptr()),                                               \
          static_cast<__nv_bfloat16 *>(output_q.data_ptr()),                                      \
          static_cast<float *>(output_s.data_ptr()),                                              \
          group_size,                                                                             \
          num_groups,                                                                             \
          groups_per_block,                                                                       \
          (float)eps,                                                                             \
          (float)min_4bit,                                                                        \
          (float)max_4bit);                                                                       \         
  } while (0)

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_UINT8(input.scalar_type(), scalar_t, [&] {
    if (dst_type == at::ScalarType::BFloat16) {
      // QUInt8
      LAUNCH_KERNEL(scalar_t, __nv_bfloat16);
      return true;
    } 

    return false;
  });


#undef LAUNCH_KERNEL
}

}


