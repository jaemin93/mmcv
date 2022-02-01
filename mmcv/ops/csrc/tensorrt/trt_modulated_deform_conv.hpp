#ifndef TRT_MODULATED_DEFORM_CONV_HPP
#define TRT_MODULATED_DEFORM_CONV_HPP
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "trt_plugin_helper.hpp"

class ModulatedDeformableConvPluginDynamic
    : public nvinfer1::IPluginV2DynamicExt {
 public:
  ModulatedDeformableConvPluginDynamic(const std::string &name,
                                       const nvinfer1::Dims stride,
                                       const nvinfer1::Dims padding,
                                       const nvinfer1::Dims dilation,
                                       const int deformableGroup,
                                       const int group);

  ModulatedDeformableConvPluginDynamic(const std::string name, const void *data,
                                       size_t length);

  ModulatedDeformableConvPluginDynamic() = delete;

  ~ModulatedDeformableConvPluginDynamic();

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
      nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) noexcept override;
  void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                       nvinfer1::IGpuAllocator *gpuAllocator) noexcept override;
  void detachFromContext() noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *pluginNamespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;

 private:
  const std::string mLayerName;
  std::string mNamespace;

  nvinfer1::Dims mStride;
  nvinfer1::Dims mPadding;
  nvinfer1::Dims mDilation;
  int mDeformableGroup;
  int mGroup;
  bool mWithBias;

  cublasHandle_t m_cublas_handle;

};

class ModulatedDeformableConvPluginDynamicCreator
    : public nvinfer1::IPluginCreator {
 public:
  ModulatedDeformableConvPluginDynamicCreator();

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;

  nvinfer1::IPluginV2 *createPlugin(
      const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                         const void *serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override;

  const char *getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
#endif  // TRT_MODULATED_DEFORM_CONV_HPP
