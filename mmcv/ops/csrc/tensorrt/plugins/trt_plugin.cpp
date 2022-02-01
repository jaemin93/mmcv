// Copyright (c) OpenMMLab. All rights reserved
#include "trt_plugin.hpp"

#include "trt_modulated_deform_conv.hpp"

REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
