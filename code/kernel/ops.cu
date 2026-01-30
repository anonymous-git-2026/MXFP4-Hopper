
#include "per_token_quant_mxfp4.h"
#include "per_token_Mxfp4ToBf16_dequant.h"
#include "per_token_Mxfp4ToBf16ToFp8.h"
#include "per_token_quant_mxfp4_fused_ToFp8.h"
#include "per_token_Mxfp4RowToBf16ToFp8Col.h"
#include "per_token_Mxfp4ToFp8_col.h"
#include "per_token_Mxfp4RowToBf16ToFp8Col_msplit.h"


#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }
namespace omo {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("omo_per_token_group_quant_fp4", &omo_per_token_group_quant_fp4, "Per-token-group FP4 quantization",
        py::arg("input"), py::arg("output_q"), py::arg("output_s"),py::arg("group_size"), py::arg("eps"), py::arg("fp4_min"),py::arg("fp4_max"), py::arg("scale_ue8m0"));
  m.def("omo_per_token_Mxfp4ToBf16_dequant", &omo_per_token_Mxfp4ToBf16_dequant, "Per-token-group FP4 dequantization BF16",
        py::arg("input"), py::arg("input_s"),py::arg("output_q"), py::arg("output_s"),py::arg("group_size"), py::arg("eps"), py::arg("fp4_min"),py::arg("fp4_max"), py::arg("scale_ue8m0"));
  m.def("omo_per_token_Mxfp4ToBf16ToFp8", &omo_per_token_Mxfp4ToBf16ToFp8, "Per-token-group FP4 dequantization BF16->FP8",
        py::arg("input"), py::arg("input_s"),py::arg("output_q"), py::arg("output_s"),py::arg("group_size"), py::arg("eps"), py::arg("fp4_min"),py::arg("fp4_max"), py::arg("scale_ue8m0"));
  m.def("omo_per_token_group_quant_fp4_fused_ToFp8", &omo_per_token_group_quant_fp4_fused_ToFp8, "Per-token-group FP4 dequantization fused ToFp8",
        py::arg("input"), py::arg("output_q"), py::arg("output_s"),py::arg("output_bf16_q"), py::arg("output_bf16_s"),py::arg("group_size"), py::arg("eps"), py::arg("fp4_min"),py::arg("fp4_max"), py::arg("scale_ue8m0"));
  m.def("omo_per_token_Mxfp4RowToBf16ToFp8Col", &omo_per_token_Mxfp4RowToBf16ToFp8Col, "Per-token-group FP4(row) dequantization BF16->FP8(col)",
        py::arg("input"), py::arg("input_s"),py::arg("output_q"), py::arg("output_s"), py::arg("eps"));
  m.def("omo_per_token_Mxfp4RowToBf16ToFp8Col_msplit", &omo_per_token_Mxfp4RowToBf16ToFp8Col_msplit, "Per-token-group FP4 dequantization col FP8 for group linear(m_split)",
        py::arg("input"), py::arg("input_s"),py::arg("output_q"), py::arg("output_s"), py::arg("m_splits"), py::arg("split_offsets"), py::arg("scale_offsets"), py::arg("max_split_len"), py::arg("eps"));
  m.def("omo_per_token_Mxfp4ToFp8_col", &omo_per_token_Mxfp4ToFp8_col, "Per-token-group FP4 dequantization col FP8",
        py::arg("input"), py::arg("input_s"),py::arg("output_q"), py::arg("output_s"),py::arg("group_size"), py::arg("eps"), py::arg("fp4_min"),py::arg("fp4_max"), py::arg("scale_ue8m0"));


};


}