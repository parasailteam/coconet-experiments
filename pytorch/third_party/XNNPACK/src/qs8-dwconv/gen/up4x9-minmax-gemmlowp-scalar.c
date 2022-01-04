// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up4x9__scalar(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const int32_t vmultiplier = params->gemmlowp_scalar.multiplier;
  const int64_t vq31rounding = INT64_C(0x40000000);
  const int32_t vremainder_mask = params->gemmlowp_scalar.remainder_mask;
  const uint32_t vshift = params->gemmlowp_scalar.shift;
  const int32_t vremainder_threshold = params->gemmlowp_scalar.remainder_threshold;
  const int32_t vout_min = params->gemmlowp_scalar.output_min_less_zero_point;
  const int32_t vout_max = params->gemmlowp_scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->gemmlowp_scalar.output_zero_point;
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 4; c -= 4) {
      int32_t vacc0 = ((const int32_t*) w)[0];
      int32_t vacc1 = ((const int32_t*) w)[1];
      int32_t vacc2 = ((const int32_t*) w)[2];
      int32_t vacc3 = ((const int32_t*) w)[3];


      const int32_t vi0x0 = i0[0];
      const int32_t vi0x1 = i0[1];
      const int32_t vi0x2 = i0[2];
      const int32_t vi0x3 = i0[3];
      i0 += 4;

      const int32_t vk0x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[0];
      const int32_t vk0x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[1];
      const int32_t vk0x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[2];
      const int32_t vk0x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[3];

      vacc0 += vi0x0 * vk0x0;
      vacc1 += vi0x1 * vk0x1;
      vacc2 += vi0x2 * vk0x2;
      vacc3 += vi0x3 * vk0x3;

      const int32_t vi1x0 = i1[0];
      const int32_t vi1x1 = i1[1];
      const int32_t vi1x2 = i1[2];
      const int32_t vi1x3 = i1[3];
      i1 += 4;

      const int32_t vk1x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[4];
      const int32_t vk1x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[5];
      const int32_t vk1x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[6];
      const int32_t vk1x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[7];

      vacc0 += vi1x0 * vk1x0;
      vacc1 += vi1x1 * vk1x1;
      vacc2 += vi1x2 * vk1x2;
      vacc3 += vi1x3 * vk1x3;

      const int32_t vi2x0 = i2[0];
      const int32_t vi2x1 = i2[1];
      const int32_t vi2x2 = i2[2];
      const int32_t vi2x3 = i2[3];
      i2 += 4;

      const int32_t vk2x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[8];
      const int32_t vk2x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[9];
      const int32_t vk2x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[10];
      const int32_t vk2x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[11];

      vacc0 += vi2x0 * vk2x0;
      vacc1 += vi2x1 * vk2x1;
      vacc2 += vi2x2 * vk2x2;
      vacc3 += vi2x3 * vk2x3;

      const int32_t vi3x0 = i3[0];
      const int32_t vi3x1 = i3[1];
      const int32_t vi3x2 = i3[2];
      const int32_t vi3x3 = i3[3];
      i3 += 4;

      const int32_t vk3x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[12];
      const int32_t vk3x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[13];
      const int32_t vk3x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[14];
      const int32_t vk3x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[15];

      vacc0 += vi3x0 * vk3x0;
      vacc1 += vi3x1 * vk3x1;
      vacc2 += vi3x2 * vk3x2;
      vacc3 += vi3x3 * vk3x3;

      const int32_t vi4x0 = i4[0];
      const int32_t vi4x1 = i4[1];
      const int32_t vi4x2 = i4[2];
      const int32_t vi4x3 = i4[3];
      i4 += 4;

      const int32_t vk4x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[16];
      const int32_t vk4x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[17];
      const int32_t vk4x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[18];
      const int32_t vk4x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[19];

      vacc0 += vi4x0 * vk4x0;
      vacc1 += vi4x1 * vk4x1;
      vacc2 += vi4x2 * vk4x2;
      vacc3 += vi4x3 * vk4x3;

      const int32_t vi5x0 = i5[0];
      const int32_t vi5x1 = i5[1];
      const int32_t vi5x2 = i5[2];
      const int32_t vi5x3 = i5[3];
      i5 += 4;

      const int32_t vk5x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[20];
      const int32_t vk5x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[21];
      const int32_t vk5x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[22];
      const int32_t vk5x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[23];

      vacc0 += vi5x0 * vk5x0;
      vacc1 += vi5x1 * vk5x1;
      vacc2 += vi5x2 * vk5x2;
      vacc3 += vi5x3 * vk5x3;

      const int32_t vi6x0 = i6[0];
      const int32_t vi6x1 = i6[1];
      const int32_t vi6x2 = i6[2];
      const int32_t vi6x3 = i6[3];
      i6 += 4;

      const int32_t vk6x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[24];
      const int32_t vk6x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[25];
      const int32_t vk6x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[26];
      const int32_t vk6x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[27];

      vacc0 += vi6x0 * vk6x0;
      vacc1 += vi6x1 * vk6x1;
      vacc2 += vi6x2 * vk6x2;
      vacc3 += vi6x3 * vk6x3;

      const int32_t vi7x0 = i7[0];
      const int32_t vi7x1 = i7[1];
      const int32_t vi7x2 = i7[2];
      const int32_t vi7x3 = i7[3];
      i7 += 4;

      const int32_t vk7x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[28];
      const int32_t vk7x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[29];
      const int32_t vk7x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[30];
      const int32_t vk7x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[31];

      vacc0 += vi7x0 * vk7x0;
      vacc1 += vi7x1 * vk7x1;
      vacc2 += vi7x2 * vk7x2;
      vacc3 += vi7x3 * vk7x3;

      const int32_t vi8x0 = i8[0];
      const int32_t vi8x1 = i8[1];
      const int32_t vi8x2 = i8[2];
      const int32_t vi8x3 = i8[3];
      i8 += 4;

      const int32_t vk8x0 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[32];
      const int32_t vk8x1 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[33];
      const int32_t vk8x2 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[34];
      const int32_t vk8x3 = ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[35];

      vacc0 += vi8x0 * vk8x0;
      vacc1 += vi8x1 * vk8x1;
      vacc2 += vi8x2 * vk8x2;
      vacc3 += vi8x3 * vk8x3;

      w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t) + 36 * sizeof(int8_t));

      const int64_t vproduct0 = (int64_t) vacc0 * (int64_t) vmultiplier;
      const int64_t vproduct1 = (int64_t) vacc1 * (int64_t) vmultiplier;
      const int64_t vproduct2 = (int64_t) vacc2 * (int64_t) vmultiplier;
      const int64_t vproduct3 = (int64_t) vacc3 * (int64_t) vmultiplier;

      const int32_t vq31product0 = (int32_t) (uint32_t) ((uint64_t) (vproduct0 + vq31rounding) >> 31);
      const int32_t vq31product1 = (int32_t) (uint32_t) ((uint64_t) (vproduct1 + vq31rounding) >> 31);
      const int32_t vq31product2 = (int32_t) (uint32_t) ((uint64_t) (vproduct2 + vq31rounding) >> 31);
      const int32_t vq31product3 = (int32_t) (uint32_t) ((uint64_t) (vproduct3 + vq31rounding) >> 31);

      const int32_t vremainder0 = (vq31product0 & vremainder_mask) - (int32_t) (vq31product0 < 0);
      const int32_t vremainder1 = (vq31product1 & vremainder_mask) - (int32_t) (vq31product1 < 0);
      const int32_t vremainder2 = (vq31product2 & vremainder_mask) - (int32_t) (vq31product2 < 0);
      const int32_t vremainder3 = (vq31product3 & vremainder_mask) - (int32_t) (vq31product3 < 0);

      int32_t vout0 = asr_s32(vq31product0, vshift) + (int32_t) (vremainder0 > vremainder_threshold);
      int32_t vout1 = asr_s32(vq31product1, vshift) + (int32_t) (vremainder1 > vremainder_threshold);
      int32_t vout2 = asr_s32(vq31product2, vshift) + (int32_t) (vremainder2 > vremainder_threshold);
      int32_t vout3 = asr_s32(vq31product3, vshift) + (int32_t) (vremainder3 > vremainder_threshold);

      vout0 = math_max_s32(vout0, vout_min);
      vout1 = math_max_s32(vout1, vout_min);
      vout2 = math_max_s32(vout2, vout_min);
      vout3 = math_max_s32(vout3, vout_min);

      vout0 = math_min_s32(vout0, vout_max);
      vout1 = math_min_s32(vout1, vout_max);
      vout2 = math_min_s32(vout2, vout_max);
      vout3 = math_min_s32(vout3, vout_max);

      vout0 += voutput_zero_point;
      vout1 += voutput_zero_point;
      vout2 += voutput_zero_point;
      vout3 += voutput_zero_point;

      output[0] = (int8_t) vout0;
      output[1] = (int8_t) vout1;
      output[2] = (int8_t) vout2;
      output[3] = (int8_t) vout3;
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t));
      do {
        int32_t vacc = *((const int32_t*) w);
        w = (const void*) ((uintptr_t) w + sizeof(int32_t));

        const int32_t vi0 = *i0++;
        const int32_t vk0 = k[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = *i1++;
        const int32_t vk1 = k[4];
        vacc += vi1 * vk1;
        const int32_t vi2 = *i2++;
        const int32_t vk2 = k[8];
        vacc += vi2 * vk2;
        const int32_t vi3 = *i3++;
        const int32_t vk3 = k[12];
        vacc += vi3 * vk3;
        const int32_t vi4 = *i4++;
        const int32_t vk4 = k[16];
        vacc += vi4 * vk4;
        const int32_t vi5 = *i5++;
        const int32_t vk5 = k[20];
        vacc += vi5 * vk5;
        const int32_t vi6 = *i6++;
        const int32_t vk6 = k[24];
        vacc += vi6 * vk6;
        const int32_t vi7 = *i7++;
        const int32_t vk7 = k[28];
        vacc += vi7 * vk7;
        const int32_t vi8 = *i8++;
        const int32_t vk8 = k[32];
        vacc += vi8 * vk8;
        k += 1;

        const int64_t vproduct = (int64_t) vacc * (int64_t) vmultiplier;
        const int32_t vq31product = (int32_t) (uint32_t) ((uint64_t) (vproduct + vq31rounding) >> 31);
        const int32_t vremainder = (vq31product & vremainder_mask) - (int32_t) (vq31product < 0);

        int32_t vout = asr_s32(vq31product, vshift) + (int32_t) (vremainder > vremainder_threshold);
        vout = math_max_s32(vout, vout_min);
        vout = math_min_s32(vout, vout_max);
        vout += voutput_zero_point;
        *output++ = (int8_t) vout;
      } while (--c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
