#pragma once
#include <torch/extension.h>
#include <torch/types.h>

namespace omo {
  void omo_per_token_Mxfp4RowToBf16ToFp8Col(at::Tensor input, at::Tensor input_s,at::Tensor output_q, at::Tensor output_s, float eps);
}  // namespace omo