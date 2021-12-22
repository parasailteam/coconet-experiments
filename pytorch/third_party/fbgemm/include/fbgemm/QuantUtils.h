#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include "./FbgemmBuild.h"
#include "./QuantUtilsAvx2.h"
#include "./Utils.h"

namespace fbgemm {

FBGEMM_API TensorQuantizationParams ChooseQuantizationParams(
    float min,
    float max,
    std::int32_t qmin,
    std::int32_t qmax,
    bool preserve_sparsity = false,
    bool force_scale_power_of_two = false);

FBGEMM_API void ChooseRequantizationMultiplier(
    float real_multiplier,
    std::int32_t* quantized_multiplier,
    int* right_shift,
    int requantization_multiplier_precision = 32);

////////////////////////////////////////////////////////////////////////////////
// Utility functions

// Clamp src in T1 to the desired precision and convert it to T2
// TODO: T26263653 fix signed-integer-overflow undefined behavior
template <typename T1, typename T2 = std::uint8_t>
NO_SANITIZE("signed-integer-overflow")
T2 clamp(T1 src, int precision, bool is_signed = false) {
  std::int32_t min = is_signed ? -(1LL << (precision - 1)) : 0;
  std::int32_t max =
      is_signed ? ((1LL << (precision - 1)) - 1) : (1LL << precision) - 1;

  // Make sure T1 and T2 can represent the precision
  assert(min >= std::numeric_limits<T1>::lowest());
  assert(min >= std::numeric_limits<T2>::lowest());
  assert(max <= std::numeric_limits<T1>::max());
  assert(max <= std::numeric_limits<T2>::max());

  return std::min<T1>(std::max<T1>(src, min), max);
}

/// Quantize src using zero_point and scale, clamp to the specified precision,
/// and convert it to type T
template <typename T>
T Quantize(
    float src,
    std::int32_t zero_point,
    float scale,
    int result_precision,
    bool result_is_signed = std::is_signed<T>::value) {
  const float transformed_val = zero_point + src / scale;
  // Please note the use of double. Unlike float, a double can represent
  // all int32 values exactly. Using a float results in a float value >
  // INT32_MAX conversion to int32 in clamp function and hence an UBSAN error.
  return clamp<double, T>(
      std::nearbyint(transformed_val), result_precision, result_is_signed);
}

template <typename T>
T Quantize(float src, const TensorQuantizationParams& qparams) {
  return Quantize<T>(src, qparams.zero_point, qparams.scale, qparams.precision);
}

template <typename T>
FBGEMM_API void Quantize(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams,
    int thread_id = 0,
    int num_threads = 1);

/*
 * @brief Quantize floating point data in src to type T
 *
 * @tparam T output quantized data type (int8_t, uint8_t and int32_t are
 *                  supported)
 *
 * @tparam T LAYOUT layout of input tensor in src. (KCX and KXC are supported)
 *                  KCX corresponds to KCRS or KCTRS (for weight tensors with
 *                  time dimension)
 *                  KXC corresponds to KRSC or KTRSC (for weight tensors with
 *                  time dimension)
 *
 * @param K Output channels for weight tensors
 * @param C Number of channels
 * @param X R*S or T*R*S
 * @param G Groups (if G == C the function performs channelwise quantization;
 *                  if 1 < G < C the function performs groupwise quantization;
 *                  if G == 1 the function performs per tensor quantization;)
 * @param scales floating point scales.
 *               Size should be equal G
 * @param zero_points zero points (should be reprsentable in type T).
 *                    Size should be equal G
 */
template <typename T, layout_t LAYOUT = layout_t::KCX>
FBGEMM_API void QuantizeGroupwise(
    const float* src,
    int K,
    int C,
    int X,
    int G,
    const float* scales,
    const std::int32_t* zero_points,
    T* dst);

template <typename T>
float Dequantize(T src, const TensorQuantizationParams& qparams) {
  return qparams.scale * (src - qparams.zero_point);
}

template <typename T>
void Dequantize(
    const T* src,
    float* dst,
    int len,
    const TensorQuantizationParams& qparams,
    int thread_id = 0,
    int num_threads = 1) {
  int i_begin, i_end;
  fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);
  for (std::size_t i = i_begin; i < i_end; i++) {
    dst[i] = Dequantize(src[i], qparams);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Requantization (pure fixed-point)

FBGEMM_API std::int64_t
SaturatingRoundingMulWithShift(std::int32_t a, std::int32_t b, int right_shift);

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    std::int32_t zero_point,
    std::int32_t multiplier,
    int right_shift,
    int result_precision,
    bool result_is_signed = false) {
  std::int64_t quantized_down =
      zero_point + SaturatingRoundingMulWithShift(src, multiplier, right_shift);
  return clamp<std::int64_t, T>(
      quantized_down, result_precision, result_is_signed);
}

template <typename T>
T RequantizeFixedPoint(
    std::int32_t src, // int32 input before requantization
    const RequantizationParams& params) {
  return Requantize<T>(
      src,
      params.target_qparams.zero_point,
      params.multiplier,
      params.right_shift,
      params.target_qparams.precision);
}

template <typename T>
FBGEMM_API void RequantizeFixedPoint(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params,
    int thread_id = 0,
    int num_threads = 1);

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    std::int32_t zero_point,
    float multiplier,
    int result_precision,
    bool result_is_signed = false) {
  long quantized_down = zero_point + std::lrintf(src * multiplier);
  return clamp<long, T>(quantized_down, result_precision, result_is_signed);
}

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    const RequantizationParams& params) {
  return Requantize<T>(
      src,
      params.target_qparams.zero_point,
      params.real_multiplier,
      params.target_qparams.precision);
}

template <typename T>
FBGEMM_API void Requantize(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params,
    int thread_id = 0,
    int num_threads = 1);

} // namespace fbgemm
