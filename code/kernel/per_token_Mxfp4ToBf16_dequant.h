#pragma once
#include <torch/extension.h>
#include <torch/types.h>

namespace omo {
  void omo_per_token_Mxfp4ToBf16_dequant(at::Tensor input, at::Tensor input_s,at::Tensor output_q, at::Tensor output_s, int group_size,float eps, float fp4_min, float fp4_max, bool scale_ue8m0);

}  // namespace omo