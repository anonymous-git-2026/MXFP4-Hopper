from omo import omo_per_token_Mxfp4ToBf16ToFp8,omo_per_token_Mxfp4ToBf16_dequant,omo_per_token_Mxfp4ToFp8,omo_per_token_Mxfp4ToFp8_col
import torch
import itertools
from typing import Optional, Tuple


fp8_type_ =  torch.float8_e4m3fn
uint8_type = torch.uint8
fp8_max = torch.finfo(fp8_type_).max
fp8_min = -fp8_max
mxfp4_max = 6.0
mxfp4_min = -mxfp4_max
import torch.nn.functional as F

def pad_to_multiple(x, multiple=128):

    h, w = x.shape[-2:]
    last_dim = x.shape[-1]
    

    pad_amount = (multiple - (last_dim % multiple)) % multiple

    if pad_amount > 0:
        x_padded = F.pad(x, (0, pad_amount))
        # print("padding triggered")
    else:
        x_padded = x
        
    return x_padded, last_dim

def unpad_from_multiple(x_padded, original_size):

    return x_padded[..., :original_size]


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

def omo_per_token_mxfp4ToBf16_dequant_wrapper(
    x: torch.Tensor,
    x_in_s:torch.Tensor,
    dtype
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
    out_shape = (*x.shape[:-1], x.shape[-1] * 2)
    scale_shape = (*x.shape[:-1], x.shape[-1] * 2 // 128)


    x_q = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
    x_s = torch.empty(scale_shape, device=x.device, dtype=torch.float)
    # x_q = x.new_empty(out_shape,dtype=uint8_type)

    # print(f"x_s{x_s.shape}")

    if x.shape[0] > 0:
        # Temporary
        
        omo_per_token_Mxfp4ToBf16_dequant(
            x, x_in_s,x_q, x_s, group_size, eps, mxfp4_min, mxfp4_max, scale_ue8m0
        )

    return x_q.to(dtype)




def omo_per_token_mxfp4ToBf16ToFp8_wrapper(
    x: torch.Tensor,
    x_in_s:torch.Tensor,
):
    group_size = 128
    dst_dtype = fp8_type_
    eps = 1e-10
    column_major_scales = False
    scale_tma_aligned = True
    scale_ue8m0 = False
    
    x_padded, original_w = pad_to_multiple(x, multiple=16)

    # assert (
    #     x.shape[-1] % group_size == 0
    # ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    # out_shape = (*x.shape[:-1], x.shape[-1] * 2)
    # scale_shape = (*x.shape[:-1], (x.shape[-1] * 2  + 127) // 128)

    out_shape = (*x_padded.shape[:-1], x_padded.shape[-1] * 2)
    # scale_shape = (*x_padded.shape[:-1], (x_padded.shape[-1] * 2 // 128))
    # TE formant
    scale_shape = ((x_padded.shape[-1] * 2 // 128),*x_padded.shape[:-1])


    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_type_)
    x_s = torch.empty(scale_shape, device=x.device, dtype=torch.float)
    # x_q = x.new_empty(out_shape,dtype=uint8_type)

    # print(f"x_s{x_s.shape}")

    if x.shape[0] > 0:
        # Temporary
        
        omo_per_token_Mxfp4ToBf16ToFp8(
            x_padded, x_in_s,x_q, x_s, group_size, eps, mxfp4_min, mxfp4_max, scale_ue8m0
        )
    # x_q = unpad_from_multiple(x_q,original_w * 2)
    x_q = x_q[:,:original_w*2]
    return x_q, x_s
def omo_per_token_mxfp4ToFp8_wrapper(
    x: torch.Tensor,
    x_in_s:torch.Tensor,
    m_splits: list[int] = None
):
    group_size = 128
    dst_dtype = fp8_type_
    eps = 1e-10
    column_major_scales = False
    scale_tma_aligned = True
    scale_ue8m0 = False
    
    # assert (
    #     x.shape[-1] % group_size == 0
    # ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"
    x_padded, original_w = pad_to_multiple(x, multiple=64)


    x_in_s_pad, original_w_scale = pad_to_multiple(x_in_s, multiple=4)



    # out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    # out_shape = (*x.shape[:-1], x.shape[-1] * 2)
    out_shape = (*x_padded.shape[:-1], x_padded.shape[-1] * 2)

    # scale_shape = (*x.shape[:-1], (x.shape[-1] * 2 + 127)  // 128)
    # TE formant
    scale_shape = ((x_padded.shape[-1] * 2 // 128),*x_padded.shape[:-1])





    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_type_)
    x_s = torch.empty(scale_shape, device=x.device, dtype=torch.float)

    # x_q = x.new_empty(out_shape,dtype=uint8_type)

    # print(f"x_s{x_s.shape}")

    if x.shape[0] > 0:
        # Temporary
        
        omo_per_token_Mxfp4ToFp8(
            x_padded, x_in_s_pad,x_q, x_s, group_size, eps, mxfp4_min, mxfp4_max, scale_ue8m0
        )

    
    x_q = x_q[:,:original_w*2]
    x_s = x_s[:original_w_scale,:]
    # m_splits_scale = [(i+127)//128 for i in m_splits]
    # print(f"x_q.shape{x_q.shape}")

    # x_q = torch.split(x_q, m_splits, dim=0)
    # x_s = torch.split(x_s, m_splits, dim=1)

    x_q_split = torch.split(x_q, m_splits, dim=0)
    x_s_split = torch.split(x_s, m_splits, dim=1)




    # return x_q, x_s
    return x_q_split, x_s_split

def omo_per_token_mxfp4ToFp8_col_wrapper(
    x: torch.Tensor,
    x_in_s:torch.Tensor,
):
    group_size = 128
    dst_dtype = fp8_type_
    eps = 1e-10
    column_major_scales = False
    scale_tma_aligned = True
    scale_ue8m0 = False
    
    x_padded, original_w = pad_to_multiple(x, multiple=16)
    # print(f"x.shape:{x.shape}")
    # print(f"x_in_s.shape:{x_in_s.shape}")

    # print(f"x.shape={x.shape}")
    # assert (
    #     x.shape[-1] % group_size == 0
    # ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    # Now we need transpose output
    out_shape = (x_padded.shape[-1] * 2, *x_padded.shape[:-1])

    # print(f"x_padded.shape={x_padded.shape[0]}")
    # scale_shape = (x_padded.shape[-1] * 2, (x_padded.shape[0] //128))

    # for TE scale format
    scale_shape = ((x_padded.shape[0] //128),x_padded.shape[-1] * 2)


    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_type_)
    x_s = torch.empty(scale_shape, device=x.device, dtype=torch.float)
    # print(f"x_q.shape:{x_q.shape}")
    # print(f"x_s.shape:{x_s.shape}")


    # x_q = x.new_empty(out_shape,dtype=uint8_type)

    # print(f"x_s{x_s.shape}")

    if x.shape[0] > 0:
        # Temporary
        
        omo_per_token_Mxfp4ToFp8_col(
            x_padded, x_in_s,x_q, x_s,  group_size, eps, mxfp4_min, mxfp4_max, scale_ue8m0
        )
    x_q = x_q[...,:original_w*2]
    
    return x_q, x_s