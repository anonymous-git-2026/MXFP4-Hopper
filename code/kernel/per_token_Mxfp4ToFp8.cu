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



__device__ __forceinline__ uint8_t GroupReduceMax(uint8_t val, const int tid) {

  val = max(val, __shfl_xor_sync(FULL_MASK, val, 4));
  val = max(val, __shfl_xor_sync(FULL_MASK, val, 2));
  val = max(val, __shfl_xor_sync(FULL_MASK, val, 1));
  return val;
}
__device__ __inline__  float scale_to_float(uint8_t scale) {
  uint32_t scale_uint32 = static_cast<uint32_t>(scale) << 23;
  float scale_output;
  std::memcpy(&scale_output, &scale_uint32, sizeof(float));
  return scale_output;
}


template <
    typename T,
    typename DST_DTYPE = __nv_fp8_e4m3,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void omo_per_token_Mxfp4ToFp8_kernel(
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

  // unit8:vec_size = 16
  constexpr uint32_t vec_size_in = 8 / sizeof(T);
  constexpr uint32_t vec_size_scale_in = 8 / sizeof(T);
  constexpr uint32_t vec_size_out = 16 / sizeof(T);

  // const uint32_t group_num_per_block = 8;
  const  int TILE_DIM = 16;
  __shared__ uint8_t scale_in[8 * 16 * 8 * 2 / 32]; 
  __shared__ float shared_tile[16];

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
  // const int64_t block_group_offset_in = global_group_id_in * group_size;
  const int64_t block_group_offset_out = global_group_id_out * group_size;


  float local_absmax = eps;
  uint8_t local_absmax_uint8 = eps;
  uint16_t global_absmax_uint8 = eps;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  // static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);


  const T* group_input = input + block_group_offset_in;
  // DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset_out;

  float* scale_output = static_cast<float*>(output_s) + global_group_id_in;

  

  vec_t input_vec;
  vec_t_out out_vec;
  input_vec.cast_load(group_input + mxfp4_lane_id * vec_size_in);
  scale_in[threadIdx.x / 2] = input_s[blockIdx.x  * blockDim.x / 2 + threadIdx.x / 2];

  
  local_absmax_uint8 = (scale_in[threadIdx.x / 2]);


  global_absmax_uint8 = GroupReduceMax(local_absmax_uint8,lane_id) - 6;
  uint16_t scale_adjust = global_absmax_uint8 - local_absmax_uint8;


  


  float y_s = scale_to_float(global_absmax_uint8);

 
  const int idx = blockIdx.x * groups_per_block + threadIdx.x / 8;
  const int row = idx / (K / 64);
  const int col = idx % (K / 64);
  if(idx < M * K * 2 / 128){
    static_cast<float*>(output_s)[col * M + row] = y_s;
  }
  




  __syncthreads();

  const uint32_t output_base = mxfp4_lane_id_out * vec_size_out;
  
  #pragma unroll
  for (uint32_t j = 0; j < vec_size_in; j = j + 1) {
        uint8_t val  = static_cast<uint8_t>(input_vec[j]);
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


        uint16_t offset = static_cast<uint16_t>(scale_adjust) << dst_m_bits;
        uint16_t adjusted_0 = x0 - offset;
        uint16_t adjusted_1 = x1 - offset;
        float q_val_0 = to_float(__ushort_as_bfloat16(adjusted_0));
        float q_val_1 = to_float(__ushort_as_bfloat16(adjusted_1));

        q_val_0 = fmaxf(-448, fminf(448, q_val_0));
        q_val_1 = fmaxf(-448, fminf(448, q_val_1));



        const uint8_t fp8_0 = *reinterpret_cast<const uint8_t*>(&(__nv_fp8_e4m3)(q_val_0));
        const uint8_t fp8_1 = *reinterpret_cast<const uint8_t*>(&(__nv_fp8_e4m3)(q_val_1));
        const uint16_t packed = fp8_0 | (static_cast<uint16_t>(fp8_1) << 8);
          
        reinterpret_cast<uint16_t*>(&group_output[output_base + j * 2])[0] = packed;
    }
    __syncthreads();
}


namespace omo{
void omo_per_token_Mxfp4ToFp8(
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

  int groups_per_block = 1;

  groups_per_block = calculateGroupsConstexpr(num_groups);

  auto dst_type = output_q.scalar_type();
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
      omo_per_token_Mxfp4ToFp8_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(     \
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
      // QUInt8
      LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
      return true;
    } 
  });

#undef LAUNCH_KERNEL
}

}