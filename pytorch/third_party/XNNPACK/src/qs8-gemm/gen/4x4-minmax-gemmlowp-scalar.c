// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/gemm.h>


void xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    int32_t vacc3x2 = vacc0x2;
    int32_t vacc3x3 = vacc0x3;
    w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;
      const int32_t va3 = (int32_t) *a3++;

      const int32_t vb0 = ((const int8_t*) w)[0];
      const int32_t vb1 = ((const int8_t*) w)[1];
      const int32_t vb2 = ((const int8_t*) w)[2];
      const int32_t vb3 = ((const int8_t*) w)[3];
      w = (const void*) ((uintptr_t) w + 4 * sizeof(int8_t));

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;
      vacc3x0 += va3 * vb0;
      vacc3x1 += va3 * vb1;
      vacc3x2 += va3 * vb2;
      vacc3x3 += va3 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    const int32_t vmultiplier = params->gemmlowp_scalar.multiplier;
    const int64_t vproduct0x0 = (int64_t) vacc0x0 * (int64_t) vmultiplier;
    const int64_t vproduct0x1 = (int64_t) vacc0x1 * (int64_t) vmultiplier;
    const int64_t vproduct0x2 = (int64_t) vacc0x2 * (int64_t) vmultiplier;
    const int64_t vproduct0x3 = (int64_t) vacc0x3 * (int64_t) vmultiplier;
    const int64_t vproduct1x0 = (int64_t) vacc1x0 * (int64_t) vmultiplier;
    const int64_t vproduct1x1 = (int64_t) vacc1x1 * (int64_t) vmultiplier;
    const int64_t vproduct1x2 = (int64_t) vacc1x2 * (int64_t) vmultiplier;
    const int64_t vproduct1x3 = (int64_t) vacc1x3 * (int64_t) vmultiplier;
    const int64_t vproduct2x0 = (int64_t) vacc2x0 * (int64_t) vmultiplier;
    const int64_t vproduct2x1 = (int64_t) vacc2x1 * (int64_t) vmultiplier;
    const int64_t vproduct2x2 = (int64_t) vacc2x2 * (int64_t) vmultiplier;
    const int64_t vproduct2x3 = (int64_t) vacc2x3 * (int64_t) vmultiplier;
    const int64_t vproduct3x0 = (int64_t) vacc3x0 * (int64_t) vmultiplier;
    const int64_t vproduct3x1 = (int64_t) vacc3x1 * (int64_t) vmultiplier;
    const int64_t vproduct3x2 = (int64_t) vacc3x2 * (int64_t) vmultiplier;
    const int64_t vproduct3x3 = (int64_t) vacc3x3 * (int64_t) vmultiplier;

    const int64_t vq31rounding = INT64_C(0x40000000);
    const int32_t vq31product0x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x0 + vq31rounding) >> 31);
    const int32_t vq31product0x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x1 + vq31rounding) >> 31);
    const int32_t vq31product0x2 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x2 + vq31rounding) >> 31);
    const int32_t vq31product0x3 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x3 + vq31rounding) >> 31);
    const int32_t vq31product1x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct1x0 + vq31rounding) >> 31);
    const int32_t vq31product1x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct1x1 + vq31rounding) >> 31);
    const int32_t vq31product1x2 = (int32_t) (uint32_t) ((uint64_t) (vproduct1x2 + vq31rounding) >> 31);
    const int32_t vq31product1x3 = (int32_t) (uint32_t) ((uint64_t) (vproduct1x3 + vq31rounding) >> 31);
    const int32_t vq31product2x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct2x0 + vq31rounding) >> 31);
    const int32_t vq31product2x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct2x1 + vq31rounding) >> 31);
    const int32_t vq31product2x2 = (int32_t) (uint32_t) ((uint64_t) (vproduct2x2 + vq31rounding) >> 31);
    const int32_t vq31product2x3 = (int32_t) (uint32_t) ((uint64_t) (vproduct2x3 + vq31rounding) >> 31);
    const int32_t vq31product3x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct3x0 + vq31rounding) >> 31);
    const int32_t vq31product3x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct3x1 + vq31rounding) >> 31);
    const int32_t vq31product3x2 = (int32_t) (uint32_t) ((uint64_t) (vproduct3x2 + vq31rounding) >> 31);
    const int32_t vq31product3x3 = (int32_t) (uint32_t) ((uint64_t) (vproduct3x3 + vq31rounding) >> 31);

    const int32_t vremainder_mask = params->gemmlowp_scalar.remainder_mask;
    const int32_t vremainder0x0 = (vq31product0x0 & vremainder_mask) - (int32_t) (vq31product0x0 < 0);
    const int32_t vremainder0x1 = (vq31product0x1 & vremainder_mask) - (int32_t) (vq31product0x1 < 0);
    const int32_t vremainder0x2 = (vq31product0x2 & vremainder_mask) - (int32_t) (vq31product0x2 < 0);
    const int32_t vremainder0x3 = (vq31product0x3 & vremainder_mask) - (int32_t) (vq31product0x3 < 0);
    const int32_t vremainder1x0 = (vq31product1x0 & vremainder_mask) - (int32_t) (vq31product1x0 < 0);
    const int32_t vremainder1x1 = (vq31product1x1 & vremainder_mask) - (int32_t) (vq31product1x1 < 0);
    const int32_t vremainder1x2 = (vq31product1x2 & vremainder_mask) - (int32_t) (vq31product1x2 < 0);
    const int32_t vremainder1x3 = (vq31product1x3 & vremainder_mask) - (int32_t) (vq31product1x3 < 0);
    const int32_t vremainder2x0 = (vq31product2x0 & vremainder_mask) - (int32_t) (vq31product2x0 < 0);
    const int32_t vremainder2x1 = (vq31product2x1 & vremainder_mask) - (int32_t) (vq31product2x1 < 0);
    const int32_t vremainder2x2 = (vq31product2x2 & vremainder_mask) - (int32_t) (vq31product2x2 < 0);
    const int32_t vremainder2x3 = (vq31product2x3 & vremainder_mask) - (int32_t) (vq31product2x3 < 0);
    const int32_t vremainder3x0 = (vq31product3x0 & vremainder_mask) - (int32_t) (vq31product3x0 < 0);
    const int32_t vremainder3x1 = (vq31product3x1 & vremainder_mask) - (int32_t) (vq31product3x1 < 0);
    const int32_t vremainder3x2 = (vq31product3x2 & vremainder_mask) - (int32_t) (vq31product3x2 < 0);
    const int32_t vremainder3x3 = (vq31product3x3 & vremainder_mask) - (int32_t) (vq31product3x3 < 0);

    const uint32_t vshift = params->gemmlowp_scalar.shift;
    const int32_t vremainder_threshold = params->gemmlowp_scalar.remainder_threshold;
    int32_t vout0x0 = asr_s32(vq31product0x0, vshift) + (int32_t) (vremainder0x0 > vremainder_threshold);
    int32_t vout0x1 = asr_s32(vq31product0x1, vshift) + (int32_t) (vremainder0x1 > vremainder_threshold);
    int32_t vout0x2 = asr_s32(vq31product0x2, vshift) + (int32_t) (vremainder0x2 > vremainder_threshold);
    int32_t vout0x3 = asr_s32(vq31product0x3, vshift) + (int32_t) (vremainder0x3 > vremainder_threshold);
    int32_t vout1x0 = asr_s32(vq31product1x0, vshift) + (int32_t) (vremainder1x0 > vremainder_threshold);
    int32_t vout1x1 = asr_s32(vq31product1x1, vshift) + (int32_t) (vremainder1x1 > vremainder_threshold);
    int32_t vout1x2 = asr_s32(vq31product1x2, vshift) + (int32_t) (vremainder1x2 > vremainder_threshold);
    int32_t vout1x3 = asr_s32(vq31product1x3, vshift) + (int32_t) (vremainder1x3 > vremainder_threshold);
    int32_t vout2x0 = asr_s32(vq31product2x0, vshift) + (int32_t) (vremainder2x0 > vremainder_threshold);
    int32_t vout2x1 = asr_s32(vq31product2x1, vshift) + (int32_t) (vremainder2x1 > vremainder_threshold);
    int32_t vout2x2 = asr_s32(vq31product2x2, vshift) + (int32_t) (vremainder2x2 > vremainder_threshold);
    int32_t vout2x3 = asr_s32(vq31product2x3, vshift) + (int32_t) (vremainder2x3 > vremainder_threshold);
    int32_t vout3x0 = asr_s32(vq31product3x0, vshift) + (int32_t) (vremainder3x0 > vremainder_threshold);
    int32_t vout3x1 = asr_s32(vq31product3x1, vshift) + (int32_t) (vremainder3x1 > vremainder_threshold);
    int32_t vout3x2 = asr_s32(vq31product3x2, vshift) + (int32_t) (vremainder3x2 > vremainder_threshold);
    int32_t vout3x3 = asr_s32(vq31product3x3, vshift) + (int32_t) (vremainder3x3 > vremainder_threshold);

    const int32_t vout_min = params->gemmlowp_scalar.output_min_less_zero_point;
    vout0x0 = math_max_s32(vout0x0, vout_min);
    vout0x1 = math_max_s32(vout0x1, vout_min);
    vout0x2 = math_max_s32(vout0x2, vout_min);
    vout0x3 = math_max_s32(vout0x3, vout_min);
    vout1x0 = math_max_s32(vout1x0, vout_min);
    vout1x1 = math_max_s32(vout1x1, vout_min);
    vout1x2 = math_max_s32(vout1x2, vout_min);
    vout1x3 = math_max_s32(vout1x3, vout_min);
    vout2x0 = math_max_s32(vout2x0, vout_min);
    vout2x1 = math_max_s32(vout2x1, vout_min);
    vout2x2 = math_max_s32(vout2x2, vout_min);
    vout2x3 = math_max_s32(vout2x3, vout_min);
    vout3x0 = math_max_s32(vout3x0, vout_min);
    vout3x1 = math_max_s32(vout3x1, vout_min);
    vout3x2 = math_max_s32(vout3x2, vout_min);
    vout3x3 = math_max_s32(vout3x3, vout_min);

    const int32_t vout_max = params->gemmlowp_scalar.output_max_less_zero_point;
    vout0x0 = math_min_s32(vout0x0, vout_max);
    vout0x1 = math_min_s32(vout0x1, vout_max);
    vout0x2 = math_min_s32(vout0x2, vout_max);
    vout0x3 = math_min_s32(vout0x3, vout_max);
    vout1x0 = math_min_s32(vout1x0, vout_max);
    vout1x1 = math_min_s32(vout1x1, vout_max);
    vout1x2 = math_min_s32(vout1x2, vout_max);
    vout1x3 = math_min_s32(vout1x3, vout_max);
    vout2x0 = math_min_s32(vout2x0, vout_max);
    vout2x1 = math_min_s32(vout2x1, vout_max);
    vout2x2 = math_min_s32(vout2x2, vout_max);
    vout2x3 = math_min_s32(vout2x3, vout_max);
    vout3x0 = math_min_s32(vout3x0, vout_max);
    vout3x1 = math_min_s32(vout3x1, vout_max);
    vout3x2 = math_min_s32(vout3x2, vout_max);
    vout3x3 = math_min_s32(vout3x3, vout_max);

    const int32_t voutput_zero_point = params->gemmlowp_scalar.output_zero_point;
    vout0x0 += voutput_zero_point;
    vout0x1 += voutput_zero_point;
    vout0x2 += voutput_zero_point;
    vout0x3 += voutput_zero_point;
    vout1x0 += voutput_zero_point;
    vout1x1 += voutput_zero_point;
    vout1x2 += voutput_zero_point;
    vout1x3 += voutput_zero_point;
    vout2x0 += voutput_zero_point;
    vout2x1 += voutput_zero_point;
    vout2x2 += voutput_zero_point;
    vout2x3 += voutput_zero_point;
    vout3x0 += voutput_zero_point;
    vout3x1 += voutput_zero_point;
    vout3x2 += voutput_zero_point;
    vout3x3 += voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;
      c3[0] = (int8_t) vout3x0;
      c3[1] = (int8_t) vout3x1;
      c3[2] = (int8_t) vout3x2;
      c3[3] = (int8_t) vout3x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = (int8_t) vout3x0;
        c3[1] = (int8_t) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
        c3[0] = (int8_t) vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
