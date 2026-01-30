import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# python setup.py bdist_wheel:生成wheel包
# Supported NVIDIA GPU architectures.
NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}

# TORCH_CUDA_ARCH_LIST can have one or more architectures,
# e.g. "9.0" or "7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX". Here,
# the "9.0+PTX" option asks the
# compiler to additionally include PTX code that can be runtime-compiled
# and executed on the 8.6 or newer architectures. While the PTX code will
# not give the best performance on the newer architectures, it provides
# forward compatibility.
# env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
# if env_arch_list:
#     # Let PyTorch builder to choose device to target for.
#     device_capability = ""
# else:
#     device_capability = torch.cuda.get_device_capability()
#     device_capability = f"{device_capability[0]}{device_capability[1]}"

if torch.cuda.is_available():
    device_capability = torch.cuda.get_device_capability()
    print(f"GPU计算能力: {device_capability[0]}.{device_capability[1]}")
else:
    print("CUDA不可用")

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-DENABLE_BF16", # Enable BF16 for cuda_version >= 11
    "-DENABLE_FP16", # 添加FP16支持
    "-DENABLE_FP8",  # Enable FP8 for cuda_version >= 11.8
    # "-DFLASHINFER_ENABLE_F16",
    # "-DCUTLASS_DEBUG_TRACE_LEVEL=4",
    # "-DCUTLASS_TRACE_HOST",
    # "-DCUTLASS_ENABLE_CUDA_HOST_ADAPTER=1",
]
# 90
print(f"device_capability:{device_capability[0]}{device_capability[1]}")
if device_capability:
    nvcc_flags.extend([
        f"--generate-code=arch=compute_{device_capability[0]}{device_capability[1]},code=sm_{device_capability[0]}{device_capability[1]}",
        f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability[0]}{device_capability[1]}",
        f"-arch=sm_{device_capability[0]}{device_capability[1]}a",
        # f"-DDEBUG",
        # f"-DDEBUG_SMEM",      
        # f"-DCUTLASS_DEBUG_TRACE_LEVEL=1",
        # f"-DCUTLASS_TRACE_HOST",
        
    ])

ext_modules = [
    CUDAExtension(
        # "grouped_gemm_backend",
        "omo",
        # ["csrc/ops.cu", "csrc/group_gemm.cu","csrc/per_token_quant_fp8.cu"],
        ["csrc/ops.cu", "csrc/per_token_quant_mxfp4.cu","csrc/per_token_Mxfp4ToBf16_dequant.cu","csrc/per_token_Mxfp4ToBf16ToFp8.cu","csrc/per_token_Mxfp4RowToBf16ToFp8Col.cu","csrc/per_token_Mxfp4RowToBf16ToFp8Col_msplit.cu", "csrc/per_token_Mxfp4ToFp8_col.cu"],
        include_dirs = [
            f"{cwd}/third-party/cutlass/include/",
            f"{cwd}/third-party/cutlass/tools/util/include/"
        ],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "nvcc": nvcc_flags,
        }
    )
]

setup(
    name="omoQuant",
    version="1.0.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[ "numpy", "torch"],
)