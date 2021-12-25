#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Optional.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace at {
namespace native {

// Take a Device that may not have device_index set (i.e., having it as -1
// representing the current device) and return the corresponding Device
// according to the actual device at the time of this function call.  No-op
// if the device_index is set.
static inline Device ensure_has_index(Device device) {
  if (device.is_cpu() || device.has_index()) {
    return device;
  }
  const c10::impl::DeviceGuardImplInterface* impl = c10::impl::getDeviceGuardImpl(device.type());
  return impl->getDevice();
}

static inline optional<Device> ensure_has_index(optional<Device> device) {
  if (!device.has_value()) {
    return nullopt;
  }
  return ensure_has_index(device.value());
}

Tensor _to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(!layout.has_value() || self.layout() == layout.value(),
           "to(options) doesn't support converting to a different layout, "
           "but got self.layout being ", self.layout(),
           " and options.layout set as ", layout.value());
  auto options = TensorOptions()
    .dtype(dtype)
    .layout(layout)
    .device(device)
    .pinned_memory(pin_memory);

  if (options.has_device()) {
    options = options.device(ensure_has_index(options.device()));
  }
  // memory_format is handled separately due to MemoryFormat::Preserve logic
  options = self.options().merge_in(options).memory_format(c10::nullopt);
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  bool pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() &&
                  (options.layout() == c10::kStrided));

  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense() && options.device().supports_as_strided()) {
      Tensor r;
      if (self.is_quantized()) {
        r = at::empty_quantized(self.sizes(), self, options);
        at::QuantizerPtr quantizer = r.quantizer();
        r.copy_(self, non_blocking);
        set_quantizer_(r, quantizer);
      } else {
        r = at::empty_strided(
            self.sizes(),
            self.strides(),
            options.pinned_memory(pin_out));
        r.copy_(self, non_blocking);
      }
      return r;
    } else {
      memory_format = self.suggest_memory_format();
    }
  }
  // See Note [Explicit nullopt MemoryFormat argument]
  auto r = at::empty(self.sizes(),
                     options.memory_format(memory_format).pinned_memory(pin_out),
                     c10::nullopt);
  r.copy_(self, non_blocking);
  return r;
}

template <typename T>
static inline bool is_null_or_equal_to(const c10::optional<T>& test, const T& value) {
  if (!test.has_value()) {
    return true;
  }
  return test.value() == value;
}

static inline Tensor to_impl(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format = optional_memory_format.value_or(MemoryFormat::Preserve);

  // fast path
  if (is_null_or_equal_to(dtype, self.dtype().toScalarType()) &&
      is_null_or_equal_to(layout, self.layout()) &&
      is_null_or_equal_to(device, self.device()) &&
      !copy &&
      (memory_format == MemoryFormat::Preserve ||
       self.suggest_memory_format() == memory_format)) {
    return self;
  }

  return at::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, optional_memory_format);
}

Tensor to(
  const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
  bool non_blocking,
  bool copy,
  c10::optional<c10::MemoryFormat> optional_memory_format
) {
  return to_impl(
      self,
      dtype,
      layout,
      ensure_has_index(device),
      pin_memory,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(const Tensor& self, Device device, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      nullopt,
      ensure_has_index(device),
      nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  return to_impl(
      self,
      dtype,
      nullopt,
      nullopt,
      nullopt,
      non_blocking,
      copy,
      optional_memory_format);
}

Tensor to(const Tensor& self, const Tensor& other, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto options = other.options();
  return to_impl(
      self,
      options.dtype().toScalarType(),
      options.layout(),
      options.device(),
      options.pinned_memory(),
      non_blocking,
      copy,
      optional_memory_format);
}

// This op is important primarily for lazy / graph-based backends.
// While this vanilla implementation loops through each tensor and independently converts it to cpu,
// a lazy backend like XLA might need to tell sync updates across tensors.
std::vector<Tensor> _to_cpu(TensorList tensors) {
    std::vector<Tensor> cpu_tensors;
    for (const auto& t : tensors) {
        cpu_tensors.push_back(t.cpu());
    }
    return cpu_tensors;
}

Tensor to_dense_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() != c10::kStrided);
  if (input_.layout() == c10::kSparse) {
    auto input = input_.coalesce();
    return grad.sparse_mask(input);
  } else if (input_.layout() == c10::kMkldnn) {
    return grad.to_mkldnn(input_.scalar_type());
  } else {
    AT_ERROR("Unsupported input layout: ", input_.layout());
  }
}

Tensor to_mkldnn_backward(const Tensor& grad, const Tensor& input_) {
  AT_ASSERT(input_.layout() == c10::kStrided);
  return grad.to_dense(input_.scalar_type());
}

Tensor view_dtype(const Tensor& self, ScalarType dtype) {
  if (self.scalar_type() == dtype) {
    return self;
  }
  const auto type_meta = c10::scalarTypeToTypeMeta(dtype);
  TORCH_CHECK(self.element_size() == static_cast<int64_t>(type_meta.itemsize()),
    "Viewing a tensor as a new dtype with a different number of bytes per element is not supported.");
  TORCH_CHECK(!self.is_conj(),
    "torch.Tensor.view is not supported for conjugate view tensors when converting to a different dtype.");
  TORCH_CHECK(!self.is_neg(),
    "torch.Tensor.view is not supported for tensors with negative bit set when converting to a different dtype.");
  Storage storage = self.storage();
  auto new_tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), self.key_set(), type_meta);
  auto* impl = new_tensor.unsafeGetTensorImpl();
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(self.sizes(), self.strides());
  return new_tensor;
}

}} // namespace at::native
