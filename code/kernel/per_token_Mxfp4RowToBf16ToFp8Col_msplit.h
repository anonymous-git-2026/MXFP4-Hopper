#pragma once
#include <torch/extension.h>
#include <torch/types.h>

namespace omo {
  void omo_per_token_Mxfp4RowToBf16ToFp8Col_msplit(at::Tensor input, at::Tensor input_s,at::Tensor output_q, at::Tensor output_s, at::Tensor m_splits, at::Tensor split_offsets, at::Tensor scale_offsets, int max_split_len, float eps);
}  // namespace omo