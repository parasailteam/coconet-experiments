#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

#include <c10/util/irange.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <vector>

namespace torch { namespace autograd {

namespace {

template <typename F>
void _foreach_tensor(
    F fn,
    torch::jit::Stack* stack,
    size_t stack_start,
    size_t size) {
  // Enumerate over tensors in a stack, including ones in TensorLists
  int idx_tensor = 0;
  for (const auto idx_arg : c10::irange(size)) {
    auto& ivalue = (*stack)[stack_start + idx_arg];
    if (ivalue.isTensor()) {  // true for optional tensor that has value
      const auto& tensor = ivalue.toTensor();
      fn(idx_tensor, idx_arg, tensor);
      idx_tensor++;
    } else if (ivalue.isTensorList()) {
      for (const auto& iv : ivalue.toListRef()) {
        const auto& tensor = iv.toTensor();
        fn(idx_tensor, idx_arg, tensor);
        idx_tensor++;
      }
    }
  }
}

}

void autogradNotImplementedFallbackImpl(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // Mimics the logic of a VariableType NotImplemented kernel
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto& arguments = schema.arguments();
  const auto& returns = schema.returns();
  const auto num_arguments = arguments.size();
  const auto num_returns = returns.size();
  const auto stack_start = stack->size() - num_arguments;
  const bool grad_mode = GradMode::is_enabled();
  std::vector<const at::Tensor*> tensors_requiring_grad_on_stack;

  // Keep track of which outputs are output of in-place modification
  // so we can rebase_history if necessary
  std::vector<bool> is_inplace_output;
  bool any_is_inplace_output = false;
  std::vector<bool> is_aliased_output;
  is_inplace_output.reserve(num_returns);
  is_aliased_output.reserve(num_returns);

  for (const auto i : c10::irange(num_returns)) {
    const auto& alias_info = returns[i].alias_info();
    is_inplace_output.push_back(alias_info.has_value() && alias_info->isWrite());
    any_is_inplace_output |= alias_info.has_value() && alias_info->isWrite();
    is_aliased_output.push_back(alias_info.has_value());

  }
  int aliased_input_idx = -1;
  int aliased_output_idx = -1;
  for (const auto i : c10::irange(num_returns)) {
    const auto& alias_info = returns[i].alias_info();
    if (alias_info.has_value() && !alias_info->isWrite()) {
      AT_ASSERT(
        aliased_output_idx == -1,
        "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
        "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
        "Please rewrite your function as a composite function.");
      aliased_output_idx = i;
    }
  }
  for (const auto i : c10::irange(num_arguments)) {
    const auto& alias_info = arguments[i].alias_info();
    if (alias_info.has_value() && !alias_info->isWrite()) {
      AT_ASSERT(
        aliased_input_idx == -1,
        "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
        "Non-composite functions where multiple inputs are aliased with outputs aren't supported. "
        "Please rewrite your function as a composite function.");
      aliased_input_idx = i;
    }
  }

  size_t num_tensor_inputs = 0;  // Only used for DEBUG-only checks

  _foreach_tensor([&](size_t _, size_t idx_arg, const at::Tensor& t) {
    if (grad_mode && t.requires_grad()) {
      tensors_requiring_grad_on_stack.push_back(&t);
    }
    num_tensor_inputs++;
    TORCH_CHECK_NOT_IMPLEMENTED(!isFwGradDefined(t), "Trying to use forward AD with ", op_name, " that does not support it.");
  }, stack, stack_start, num_arguments);

  const bool any_requires_grad = tensors_requiring_grad_on_stack.size() > 0;

  _foreach_tensor([&](size_t _, size_t i, const at::Tensor& t) {
    const auto& alias_info = arguments[i].alias_info();
    if (alias_info.has_value() && alias_info->isWrite()) {
      check_inplace(t, any_requires_grad);
    }
  }, stack, stack_start, num_arguments);

  std::shared_ptr<NotImplemented> grad_fn;
  if (any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented(op_name), deleteNode);
    grad_fn->set_next_edges(collect_next_edges(tensors_requiring_grad_on_stack));
  }

  #ifndef NDEBUG
  // See NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
  auto stack_args_copy = std::vector<c10::IValue>(stack->begin() + stack_start, stack->end());
  std::vector<c10::intrusive_ptr<c10::TensorImpl>> impl_saved;
  impl_saved.reserve(num_tensor_inputs);
  std::vector<c10::optional<c10::Storage>> storage_saved;
  storage_saved.reserve(num_tensor_inputs);
  _foreach_tensor([&](size_t idx, size_t _, const at::Tensor& t) {
    storage_saved.push_back(t.has_storage() ? c10::optional<c10::Storage>(t.storage()) : c10::nullopt);
    impl_saved.push_back(t.getIntrusivePtr());
  }, &stack_args_copy, 0, num_arguments);
  #endif
  if (aliased_input_idx != -1 || any_is_inplace_output) {
    at::AutoDispatchBelowAutograd guard;
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
  } else {
    // If neither in-place nor view
    at::AutoDispatchBelowADInplaceOrView guard;
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
  }
  #ifndef NDEBUG
  _foreach_tensor([&](size_t idx_tensor, size_t _, const at::Tensor& t) {
    if (storage_saved.at(idx_tensor).has_value())
      TORCH_INTERNAL_ASSERT(storage_saved.at(idx_tensor).value().is_alias_of(t.storage()), op_name);
    if (impl_saved.at(idx_tensor))
      TORCH_INTERNAL_ASSERT(impl_saved.at(idx_tensor) == t.getIntrusivePtr(), op_name);
  }, &stack_args_copy, 0, num_arguments);
  _foreach_tensor([&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
    if (!is_inplace_output[idx_ret])
      TORCH_INTERNAL_ASSERT(t.use_count() <= 1, op_name);  // Okay to return undefined tensor
    if (!is_aliased_output[idx_ret] && t.has_storage())
      TORCH_INTERNAL_ASSERT(t.storage().use_count() == 1);
  }, stack, stack->size() - num_returns, num_returns);
  // There should be only a single base-view pair, make sure their storage is aliased
  if (aliased_input_idx != -1 && aliased_output_idx != -1) {
    const c10::IValue& aliased_input_iv = stack_args_copy[aliased_input_idx];
    const c10::IValue& aliased_output_iv = (*stack)[stack->size() - num_returns + aliased_output_idx];
    // We do not support views embedded inside tensorlist
    TORCH_INTERNAL_ASSERT(aliased_input_iv.isTensor(), op_name);
    TORCH_INTERNAL_ASSERT(aliased_output_iv.isTensor(), op_name);
    const at::Tensor& aliased_input = aliased_input_iv.toTensor();
    const at::Tensor& aliased_output = aliased_input_iv.toTensor();
    if(is_aliased_output[aliased_input_idx] && aliased_input.has_storage())
      TORCH_INTERNAL_ASSERT(aliased_input.storage().is_alias_of(aliased_output.storage()), op_name);
  }
  #endif

  if (any_requires_grad) {
    _foreach_tensor([&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
      if (isDifferentiableType(t.scalar_type())) {
        if (is_inplace_output[idx_ret]) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          rebase_history(const_cast<at::Tensor&>(t), grad_fn);
        } else {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          set_history(const_cast<at::Tensor&>(t), grad_fn);
        }
      }
    }, stack, stack->size() - num_returns, num_returns);
  }
}

torch::CppFunction autogradNotImplementedFallback() {
  return torch::CppFunction::makeFromBoxedFunction<&autogradNotImplementedFallbackImpl>();
}

}} // namespace torch::autograd
