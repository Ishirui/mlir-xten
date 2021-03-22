// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#pragma once

#include <memory>
#include <vector>

#include "AIRPasses.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToLinalgPass();
void registerAIRToLinalgPass();

} // namespace air
} // namespace xilinx
