import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull

@skip_for_grayskull("Conv3d not supported on Grayskull")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("output_channels", [32])
@pytest.mark.parametrize("input_channels", [32])
@pytest.mark.parametrize("input_depth", [8])
@pytest.mark.parametrize("input_height", [8])
@pytest.mark.parametrize("input_width", [8])
@pytest.mark.parametrize("filter_depth", [3])
@pytest.mark.parametrize("filter_height", [3])
@pytest.mark.parametrize("filter_width", [3])
@pytest.mark.parametrize("stride_d", [1])
@pytest.mark.parametrize("stride_h", [1])
@pytest.mark.parametrize("stride_w", [1])
@pytest.mark.parametrize("pad_d", [1])
@pytest.mark.parametrize("pad_h", [1])
@pytest.mark.parametrize("pad_w", [1])
@pytest.mark.parametrize("is_sharded", [True])
def test_conv3d_sharded(
    device,
    batch_size,
    output_channels,
    input_channels,
    input_depth,
    input_height,
    input_width,
    filter_depth,
    filter_height,
    filter_width,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    is_sharded,
):
    torch.manual_seed(0)

    input_shape = (batch_size, input_channels, input_depth, input_height, input_width)
    weight_shape = (output_channels, input_channels, filter_depth, filter_height, filter_width)
    bias_shape = (output_channels,)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_weight_tensor = torch.randn(weight_shape, dtype=torch.bfloat16)
    torch_bias_tensor = torch.randn(bias_shape, dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.conv3d(
        torch_input_tensor,
        torch_weight_tensor,
        bias=torch_bias_tensor,
        stride=(stride_d, stride_h, stride_w),
        padding=(pad_d, pad_h, pad_w),
    )

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight_tensor = ttnn.from_torch(torch_weight_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, device=device, layout=ttnn.TILE_LAYOUT)

    if is_sharded:
        tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG)

    tt_output_tensor = ttnn.conv3d(
        input_tensor=tt_input_tensor,
        weight_tensor=tt_weight_tensor,
        bias_tensor=tt_bias_tensor,
        stride=(stride_d, stride_h, stride_w),
        padding=(pad_d, pad_h, pad_w),
    )

    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.99)