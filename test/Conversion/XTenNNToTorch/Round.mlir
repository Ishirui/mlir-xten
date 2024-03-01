// RUN: aten-opt --convert-xtennn-to-torch -split-input-file %s | FileCheck %s


func.func @round_int(%arg0: tensor<1x10xi32>) -> tensor<1x10xi32> {
    %0 = xten_nn.round %arg0 : (tensor<1x10xi32>) -> tensor<1x10xi32>
    return %0 : tensor<1x10xi32>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xi32> -> !torch.vtensor<[1,10],si32>
// CHECK:      %[[OP:.+]] = torch.aten.round %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],si32> -> !torch.vtensor<[1,10],si32>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],si32> -> tensor<1x10xi32>
}

// -----

func.func @round_f32(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = xten_nn.round %arg0 : (tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xf32> -> !torch.vtensor<[1,10],f32>
// CHECK:      %[[OP:.+]] = torch.aten.round %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],f32> -> !torch.vtensor<[1,10],f32>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],f32> -> tensor<1x10xf32>
}


// -----

func.func @round_bf16(%arg0: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
    %0 = xten_nn.round %arg0 : (tensor<1x10xbf16>) -> tensor<1x10xbf16>
    return %0 : tensor<1x10xbf16>
// CHECK-LABEL:   func.func @round_bf16
// CHECK:      %[[FROM_BUILTIN:.+]] = torch_c.from_builtin_tensor %arg0 : tensor<1x10xbf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      %[[OP:.+]] = torch.aten.round %[[FROM_BUILTIN]] : !torch.vtensor<[1,10],bf16> -> !torch.vtensor<[1,10],bf16>
// CHECK:      torch_c.to_builtin_tensor %[[OP]] : !torch.vtensor<[1,10],bf16> -> tensor<1x10xbf16>
}
