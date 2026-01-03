#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::conv3d {

void Conv3dDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to conv3d need to be on device!");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Input to conv3d must be tilized");
}

}
