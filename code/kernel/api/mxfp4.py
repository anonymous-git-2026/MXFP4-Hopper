from omo import omo_per_token_group_quant_fp4
import torch
import itertools
from typing import Optional, Tuple


fp8_type_ =  torch.float8_e4m3fn
uint8_type = torch.uint8
fp8_max = torch.finfo(fp8_type_).max
fp8_min = -fp8_max
mxfp4_max = 6.0
mxfp4_min = -mxfp4_max


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
def omo_per_token_group_quant_4bit_warpper(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    

    # if dst_dtype == torch.int8:
    #     pass

    return omo_per_token_group_quant_fp4_warpper(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
        enable_v2=enable_v2,
    )
def omo_per_token_group_quant_fp4_warpper(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)

    x_q = torch.empty(out_shape, device=x.device, dtype=uint8_type)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        # Temporary
        
        omo_per_token_group_quant_fp4(
            x, x_q, x_s, group_size, eps, mxfp4_min, mxfp4_max, scale_ue8m0
        )

    return x_q, x_s





def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = align(x_s_mn, 4)
        aligned_k = align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]

    else:
        scale_shape = (x_shape[0],x_shape[1])
        return torch.empty(
            # x_shape[:-1] + (x_shape[-1] // group_size,),
            scale_shape,
            device=device,
            dtype=torch.uint8,
        )


def omo_per_token_group_quant_mxfp4(
    x: torch.Tensor,
):
    group_size = 32
    dst_dtype = fp8_type_
    eps = 1e-10
    column_major_scales = False
    scale_tma_aligned = True
    scale_ue8m0 = False
    
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    out_shape = (*x.shape[:-1], x.shape[-1] // 2)
    scale_shape = (*x.shape[:-1], x.shape[-1] // 32)


    x_q = torch.empty(out_shape, device=x.device, dtype=uint8_type)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=scale_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )
    # print(f"x_s{x_s.shape}")

    if x.shape[0] > 0:
        # Temporary
        
        omo_per_token_group_quant_fp4(
            x, x_q, x_s, group_size, eps, mxfp4_min, mxfp4_max, scale_ue8m0
        )

    return x_q, x_s
