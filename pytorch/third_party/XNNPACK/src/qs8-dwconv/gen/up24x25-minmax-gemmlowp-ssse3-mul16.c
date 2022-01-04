// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <tmmintrin.h>

#include <xnnpack/dwconv.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up24x25__ssse3_mul16(
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
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 24; c -= 24) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 4 * sizeof(int32_t)));
      __m128i vacc89AB = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 8 * sizeof(int32_t)));
      __m128i vaccCDEF = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 12 * sizeof(int32_t)));
      __m128i vaccGHIJ = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16 * sizeof(int32_t)));
      __m128i vaccKLMN = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 20 * sizeof(int32_t)));


      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 0 * sizeof(int8_t)));
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vk0x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 8 * sizeof(int8_t)));
      const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
      const __m128i vk0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 16 * sizeof(int8_t)));
      i0 += 24;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxk0x01234567 = _mm_unpacklo_epi8(vk0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk0x01234567));
      const __m128i vxi0x89ABCDEF = _mm_unpacklo_epi8(vi0x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x89ABCDEF));
      const __m128i vxk0x89ABCDEF = _mm_unpacklo_epi8(vk0x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk0x89ABCDEF));
      const __m128i vxi0xGHIJKLMN = _mm_unpacklo_epi8(vi0xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0xGHIJKLMN));
      const __m128i vxk0xGHIJKLMN = _mm_unpacklo_epi8(vk0xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk0xGHIJKLMN));

      const __m128i vp0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vp0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);
      const __m128i vp0x89ABCDEFlo = _mm_mullo_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      const __m128i vp0x89ABCDEFhi = _mm_mulhi_epi16(vxi0x89ABCDEF, vxk0x89ABCDEF);
      const __m128i vp0xGHIJKLMNlo = _mm_mullo_epi16(vxi0xGHIJKLMN, vxk0xGHIJKLMN);
      const __m128i vp0xGHIJKLMNhi = _mm_mulhi_epi16(vxi0xGHIJKLMN, vxk0xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp0x01234567lo, vp0x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp0x01234567lo, vp0x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp0x89ABCDEFlo, vp0x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp0x89ABCDEFlo, vp0x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp0xGHIJKLMNlo, vp0xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp0xGHIJKLMNlo, vp0xGHIJKLMNhi));

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 24 * sizeof(int8_t)));
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vk1x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 32 * sizeof(int8_t)));
      const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
      const __m128i vk1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 40 * sizeof(int8_t)));
      i1 += 24;

      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxk1x01234567 = _mm_unpacklo_epi8(vk1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk1x01234567));
      const __m128i vxi1x89ABCDEF = _mm_unpacklo_epi8(vi1x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x89ABCDEF));
      const __m128i vxk1x89ABCDEF = _mm_unpacklo_epi8(vk1x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk1x89ABCDEF));
      const __m128i vxi1xGHIJKLMN = _mm_unpacklo_epi8(vi1xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1xGHIJKLMN));
      const __m128i vxk1xGHIJKLMN = _mm_unpacklo_epi8(vk1xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk1xGHIJKLMN));

      const __m128i vp1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vp1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);
      const __m128i vp1x89ABCDEFlo = _mm_mullo_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      const __m128i vp1x89ABCDEFhi = _mm_mulhi_epi16(vxi1x89ABCDEF, vxk1x89ABCDEF);
      const __m128i vp1xGHIJKLMNlo = _mm_mullo_epi16(vxi1xGHIJKLMN, vxk1xGHIJKLMN);
      const __m128i vp1xGHIJKLMNhi = _mm_mulhi_epi16(vxi1xGHIJKLMN, vxk1xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp1x01234567lo, vp1x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp1x01234567lo, vp1x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp1x89ABCDEFlo, vp1x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp1x89ABCDEFlo, vp1x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp1xGHIJKLMNlo, vp1xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp1xGHIJKLMNlo, vp1xGHIJKLMNhi));

      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 48 * sizeof(int8_t)));
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vk2x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 56 * sizeof(int8_t)));
      const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
      const __m128i vk2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 64 * sizeof(int8_t)));
      i2 += 24;

      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxk2x01234567 = _mm_unpacklo_epi8(vk2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk2x01234567));
      const __m128i vxi2x89ABCDEF = _mm_unpacklo_epi8(vi2x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x89ABCDEF));
      const __m128i vxk2x89ABCDEF = _mm_unpacklo_epi8(vk2x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk2x89ABCDEF));
      const __m128i vxi2xGHIJKLMN = _mm_unpacklo_epi8(vi2xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2xGHIJKLMN));
      const __m128i vxk2xGHIJKLMN = _mm_unpacklo_epi8(vk2xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk2xGHIJKLMN));

      const __m128i vp2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vp2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);
      const __m128i vp2x89ABCDEFlo = _mm_mullo_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      const __m128i vp2x89ABCDEFhi = _mm_mulhi_epi16(vxi2x89ABCDEF, vxk2x89ABCDEF);
      const __m128i vp2xGHIJKLMNlo = _mm_mullo_epi16(vxi2xGHIJKLMN, vxk2xGHIJKLMN);
      const __m128i vp2xGHIJKLMNhi = _mm_mulhi_epi16(vxi2xGHIJKLMN, vxk2xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp2x01234567lo, vp2x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp2x01234567lo, vp2x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp2x89ABCDEFlo, vp2x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp2x89ABCDEFlo, vp2x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp2xGHIJKLMNlo, vp2xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp2xGHIJKLMNlo, vp2xGHIJKLMNhi));

      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 72 * sizeof(int8_t)));
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vk3x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 80 * sizeof(int8_t)));
      const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
      const __m128i vk3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 88 * sizeof(int8_t)));
      i3 += 24;

      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxk3x01234567 = _mm_unpacklo_epi8(vk3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk3x01234567));
      const __m128i vxi3x89ABCDEF = _mm_unpacklo_epi8(vi3x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x89ABCDEF));
      const __m128i vxk3x89ABCDEF = _mm_unpacklo_epi8(vk3x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk3x89ABCDEF));
      const __m128i vxi3xGHIJKLMN = _mm_unpacklo_epi8(vi3xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3xGHIJKLMN));
      const __m128i vxk3xGHIJKLMN = _mm_unpacklo_epi8(vk3xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk3xGHIJKLMN));

      const __m128i vp3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vp3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);
      const __m128i vp3x89ABCDEFlo = _mm_mullo_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      const __m128i vp3x89ABCDEFhi = _mm_mulhi_epi16(vxi3x89ABCDEF, vxk3x89ABCDEF);
      const __m128i vp3xGHIJKLMNlo = _mm_mullo_epi16(vxi3xGHIJKLMN, vxk3xGHIJKLMN);
      const __m128i vp3xGHIJKLMNhi = _mm_mulhi_epi16(vxi3xGHIJKLMN, vxk3xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp3x01234567lo, vp3x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp3x01234567lo, vp3x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp3x89ABCDEFlo, vp3x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp3x89ABCDEFlo, vp3x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp3xGHIJKLMNlo, vp3xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp3xGHIJKLMNlo, vp3xGHIJKLMNhi));

      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 96 * sizeof(int8_t)));
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vk4x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 104 * sizeof(int8_t)));
      const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
      const __m128i vk4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 112 * sizeof(int8_t)));
      i4 += 24;

      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxk4x01234567 = _mm_unpacklo_epi8(vk4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk4x01234567));
      const __m128i vxi4x89ABCDEF = _mm_unpacklo_epi8(vi4x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x89ABCDEF));
      const __m128i vxk4x89ABCDEF = _mm_unpacklo_epi8(vk4x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk4x89ABCDEF));
      const __m128i vxi4xGHIJKLMN = _mm_unpacklo_epi8(vi4xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4xGHIJKLMN));
      const __m128i vxk4xGHIJKLMN = _mm_unpacklo_epi8(vk4xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk4xGHIJKLMN));

      const __m128i vp4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vp4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);
      const __m128i vp4x89ABCDEFlo = _mm_mullo_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      const __m128i vp4x89ABCDEFhi = _mm_mulhi_epi16(vxi4x89ABCDEF, vxk4x89ABCDEF);
      const __m128i vp4xGHIJKLMNlo = _mm_mullo_epi16(vxi4xGHIJKLMN, vxk4xGHIJKLMN);
      const __m128i vp4xGHIJKLMNhi = _mm_mulhi_epi16(vxi4xGHIJKLMN, vxk4xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp4x01234567lo, vp4x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp4x01234567lo, vp4x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp4x89ABCDEFlo, vp4x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp4x89ABCDEFlo, vp4x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp4xGHIJKLMNlo, vp4xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp4xGHIJKLMNlo, vp4xGHIJKLMNhi));

      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 120 * sizeof(int8_t)));
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vk5x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 128 * sizeof(int8_t)));
      const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
      const __m128i vk5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 136 * sizeof(int8_t)));
      i5 += 24;

      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxk5x01234567 = _mm_unpacklo_epi8(vk5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk5x01234567));
      const __m128i vxi5x89ABCDEF = _mm_unpacklo_epi8(vi5x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x89ABCDEF));
      const __m128i vxk5x89ABCDEF = _mm_unpacklo_epi8(vk5x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk5x89ABCDEF));
      const __m128i vxi5xGHIJKLMN = _mm_unpacklo_epi8(vi5xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5xGHIJKLMN));
      const __m128i vxk5xGHIJKLMN = _mm_unpacklo_epi8(vk5xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk5xGHIJKLMN));

      const __m128i vp5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vp5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);
      const __m128i vp5x89ABCDEFlo = _mm_mullo_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      const __m128i vp5x89ABCDEFhi = _mm_mulhi_epi16(vxi5x89ABCDEF, vxk5x89ABCDEF);
      const __m128i vp5xGHIJKLMNlo = _mm_mullo_epi16(vxi5xGHIJKLMN, vxk5xGHIJKLMN);
      const __m128i vp5xGHIJKLMNhi = _mm_mulhi_epi16(vxi5xGHIJKLMN, vxk5xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp5x01234567lo, vp5x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp5x01234567lo, vp5x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp5x89ABCDEFlo, vp5x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp5x89ABCDEFlo, vp5x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp5xGHIJKLMNlo, vp5xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp5xGHIJKLMNlo, vp5xGHIJKLMNhi));

      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 144 * sizeof(int8_t)));
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vk6x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 152 * sizeof(int8_t)));
      const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
      const __m128i vk6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 160 * sizeof(int8_t)));
      i6 += 24;

      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));
      const __m128i vxk6x01234567 = _mm_unpacklo_epi8(vk6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk6x01234567));
      const __m128i vxi6x89ABCDEF = _mm_unpacklo_epi8(vi6x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x89ABCDEF));
      const __m128i vxk6x89ABCDEF = _mm_unpacklo_epi8(vk6x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk6x89ABCDEF));
      const __m128i vxi6xGHIJKLMN = _mm_unpacklo_epi8(vi6xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6xGHIJKLMN));
      const __m128i vxk6xGHIJKLMN = _mm_unpacklo_epi8(vk6xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk6xGHIJKLMN));

      const __m128i vp6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vp6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);
      const __m128i vp6x89ABCDEFlo = _mm_mullo_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      const __m128i vp6x89ABCDEFhi = _mm_mulhi_epi16(vxi6x89ABCDEF, vxk6x89ABCDEF);
      const __m128i vp6xGHIJKLMNlo = _mm_mullo_epi16(vxi6xGHIJKLMN, vxk6xGHIJKLMN);
      const __m128i vp6xGHIJKLMNhi = _mm_mulhi_epi16(vxi6xGHIJKLMN, vxk6xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp6x01234567lo, vp6x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp6x01234567lo, vp6x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp6x89ABCDEFlo, vp6x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp6x89ABCDEFlo, vp6x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp6xGHIJKLMNlo, vp6xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp6xGHIJKLMNlo, vp6xGHIJKLMNhi));

      const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
      const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 168 * sizeof(int8_t)));
      const __m128i vi7x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i7 + 8));
      const __m128i vk7x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 176 * sizeof(int8_t)));
      const __m128i vi7xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i7 + 16));
      const __m128i vk7xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 184 * sizeof(int8_t)));
      i7 += 24;

      const __m128i vxi7x01234567 = _mm_unpacklo_epi8(vi7x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi7x01234567));
      const __m128i vxk7x01234567 = _mm_unpacklo_epi8(vk7x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk7x01234567));
      const __m128i vxi7x89ABCDEF = _mm_unpacklo_epi8(vi7x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi7x89ABCDEF));
      const __m128i vxk7x89ABCDEF = _mm_unpacklo_epi8(vk7x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk7x89ABCDEF));
      const __m128i vxi7xGHIJKLMN = _mm_unpacklo_epi8(vi7xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi7xGHIJKLMN));
      const __m128i vxk7xGHIJKLMN = _mm_unpacklo_epi8(vk7xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk7xGHIJKLMN));

      const __m128i vp7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vp7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);
      const __m128i vp7x89ABCDEFlo = _mm_mullo_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      const __m128i vp7x89ABCDEFhi = _mm_mulhi_epi16(vxi7x89ABCDEF, vxk7x89ABCDEF);
      const __m128i vp7xGHIJKLMNlo = _mm_mullo_epi16(vxi7xGHIJKLMN, vxk7xGHIJKLMN);
      const __m128i vp7xGHIJKLMNhi = _mm_mulhi_epi16(vxi7xGHIJKLMN, vxk7xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp7x01234567lo, vp7x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp7x01234567lo, vp7x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp7x89ABCDEFlo, vp7x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp7x89ABCDEFlo, vp7x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp7xGHIJKLMNlo, vp7xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp7xGHIJKLMNlo, vp7xGHIJKLMNhi));

      const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
      const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 192 * sizeof(int8_t)));
      const __m128i vi8x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i8 + 8));
      const __m128i vk8x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 200 * sizeof(int8_t)));
      const __m128i vi8xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i8 + 16));
      const __m128i vk8xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 208 * sizeof(int8_t)));
      i8 += 24;

      const __m128i vxi8x01234567 = _mm_unpacklo_epi8(vi8x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi8x01234567));
      const __m128i vxk8x01234567 = _mm_unpacklo_epi8(vk8x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk8x01234567));
      const __m128i vxi8x89ABCDEF = _mm_unpacklo_epi8(vi8x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi8x89ABCDEF));
      const __m128i vxk8x89ABCDEF = _mm_unpacklo_epi8(vk8x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk8x89ABCDEF));
      const __m128i vxi8xGHIJKLMN = _mm_unpacklo_epi8(vi8xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi8xGHIJKLMN));
      const __m128i vxk8xGHIJKLMN = _mm_unpacklo_epi8(vk8xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk8xGHIJKLMN));

      const __m128i vp8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vp8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);
      const __m128i vp8x89ABCDEFlo = _mm_mullo_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      const __m128i vp8x89ABCDEFhi = _mm_mulhi_epi16(vxi8x89ABCDEF, vxk8x89ABCDEF);
      const __m128i vp8xGHIJKLMNlo = _mm_mullo_epi16(vxi8xGHIJKLMN, vxk8xGHIJKLMN);
      const __m128i vp8xGHIJKLMNhi = _mm_mulhi_epi16(vxi8xGHIJKLMN, vxk8xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp8x01234567lo, vp8x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp8x01234567lo, vp8x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp8x89ABCDEFlo, vp8x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp8x89ABCDEFlo, vp8x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp8xGHIJKLMNlo, vp8xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp8xGHIJKLMNlo, vp8xGHIJKLMNhi));

      const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
      const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 216 * sizeof(int8_t)));
      const __m128i vi9x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i9 + 8));
      const __m128i vk9x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 224 * sizeof(int8_t)));
      const __m128i vi9xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i9 + 16));
      const __m128i vk9xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 232 * sizeof(int8_t)));
      i9 += 24;

      const __m128i vxi9x01234567 = _mm_unpacklo_epi8(vi9x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi9x01234567));
      const __m128i vxk9x01234567 = _mm_unpacklo_epi8(vk9x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk9x01234567));
      const __m128i vxi9x89ABCDEF = _mm_unpacklo_epi8(vi9x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi9x89ABCDEF));
      const __m128i vxk9x89ABCDEF = _mm_unpacklo_epi8(vk9x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk9x89ABCDEF));
      const __m128i vxi9xGHIJKLMN = _mm_unpacklo_epi8(vi9xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi9xGHIJKLMN));
      const __m128i vxk9xGHIJKLMN = _mm_unpacklo_epi8(vk9xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk9xGHIJKLMN));

      const __m128i vp9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vp9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);
      const __m128i vp9x89ABCDEFlo = _mm_mullo_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF);
      const __m128i vp9x89ABCDEFhi = _mm_mulhi_epi16(vxi9x89ABCDEF, vxk9x89ABCDEF);
      const __m128i vp9xGHIJKLMNlo = _mm_mullo_epi16(vxi9xGHIJKLMN, vxk9xGHIJKLMN);
      const __m128i vp9xGHIJKLMNhi = _mm_mulhi_epi16(vxi9xGHIJKLMN, vxk9xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp9x01234567lo, vp9x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp9x01234567lo, vp9x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp9x89ABCDEFlo, vp9x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp9x89ABCDEFlo, vp9x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp9xGHIJKLMNlo, vp9xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp9xGHIJKLMNlo, vp9xGHIJKLMNhi));

      const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
      const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 240 * sizeof(int8_t)));
      const __m128i vi10x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i10 + 8));
      const __m128i vk10x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 248 * sizeof(int8_t)));
      const __m128i vi10xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i10 + 16));
      const __m128i vk10xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 256 * sizeof(int8_t)));
      i10 += 24;

      const __m128i vxi10x01234567 = _mm_unpacklo_epi8(vi10x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi10x01234567));
      const __m128i vxk10x01234567 = _mm_unpacklo_epi8(vk10x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk10x01234567));
      const __m128i vxi10x89ABCDEF = _mm_unpacklo_epi8(vi10x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi10x89ABCDEF));
      const __m128i vxk10x89ABCDEF = _mm_unpacklo_epi8(vk10x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk10x89ABCDEF));
      const __m128i vxi10xGHIJKLMN = _mm_unpacklo_epi8(vi10xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi10xGHIJKLMN));
      const __m128i vxk10xGHIJKLMN = _mm_unpacklo_epi8(vk10xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk10xGHIJKLMN));

      const __m128i vp10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vp10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);
      const __m128i vp10x89ABCDEFlo = _mm_mullo_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);
      const __m128i vp10x89ABCDEFhi = _mm_mulhi_epi16(vxi10x89ABCDEF, vxk10x89ABCDEF);
      const __m128i vp10xGHIJKLMNlo = _mm_mullo_epi16(vxi10xGHIJKLMN, vxk10xGHIJKLMN);
      const __m128i vp10xGHIJKLMNhi = _mm_mulhi_epi16(vxi10xGHIJKLMN, vxk10xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp10x01234567lo, vp10x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp10x01234567lo, vp10x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp10x89ABCDEFlo, vp10x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp10x89ABCDEFlo, vp10x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp10xGHIJKLMNlo, vp10xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp10xGHIJKLMNlo, vp10xGHIJKLMNhi));

      const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
      const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 264 * sizeof(int8_t)));
      const __m128i vi11x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i11 + 8));
      const __m128i vk11x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 272 * sizeof(int8_t)));
      const __m128i vi11xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i11 + 16));
      const __m128i vk11xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 280 * sizeof(int8_t)));
      i11 += 24;

      const __m128i vxi11x01234567 = _mm_unpacklo_epi8(vi11x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi11x01234567));
      const __m128i vxk11x01234567 = _mm_unpacklo_epi8(vk11x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk11x01234567));
      const __m128i vxi11x89ABCDEF = _mm_unpacklo_epi8(vi11x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi11x89ABCDEF));
      const __m128i vxk11x89ABCDEF = _mm_unpacklo_epi8(vk11x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk11x89ABCDEF));
      const __m128i vxi11xGHIJKLMN = _mm_unpacklo_epi8(vi11xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi11xGHIJKLMN));
      const __m128i vxk11xGHIJKLMN = _mm_unpacklo_epi8(vk11xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk11xGHIJKLMN));

      const __m128i vp11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vp11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);
      const __m128i vp11x89ABCDEFlo = _mm_mullo_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF);
      const __m128i vp11x89ABCDEFhi = _mm_mulhi_epi16(vxi11x89ABCDEF, vxk11x89ABCDEF);
      const __m128i vp11xGHIJKLMNlo = _mm_mullo_epi16(vxi11xGHIJKLMN, vxk11xGHIJKLMN);
      const __m128i vp11xGHIJKLMNhi = _mm_mulhi_epi16(vxi11xGHIJKLMN, vxk11xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp11x01234567lo, vp11x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp11x01234567lo, vp11x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp11x89ABCDEFlo, vp11x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp11x89ABCDEFlo, vp11x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp11xGHIJKLMNlo, vp11xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp11xGHIJKLMNlo, vp11xGHIJKLMNhi));

      const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
      const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 288 * sizeof(int8_t)));
      const __m128i vi12x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i12 + 8));
      const __m128i vk12x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 296 * sizeof(int8_t)));
      const __m128i vi12xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i12 + 16));
      const __m128i vk12xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 304 * sizeof(int8_t)));
      i12 += 24;

      const __m128i vxi12x01234567 = _mm_unpacklo_epi8(vi12x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi12x01234567));
      const __m128i vxk12x01234567 = _mm_unpacklo_epi8(vk12x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk12x01234567));
      const __m128i vxi12x89ABCDEF = _mm_unpacklo_epi8(vi12x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi12x89ABCDEF));
      const __m128i vxk12x89ABCDEF = _mm_unpacklo_epi8(vk12x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk12x89ABCDEF));
      const __m128i vxi12xGHIJKLMN = _mm_unpacklo_epi8(vi12xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi12xGHIJKLMN));
      const __m128i vxk12xGHIJKLMN = _mm_unpacklo_epi8(vk12xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk12xGHIJKLMN));

      const __m128i vp12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vp12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);
      const __m128i vp12x89ABCDEFlo = _mm_mullo_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);
      const __m128i vp12x89ABCDEFhi = _mm_mulhi_epi16(vxi12x89ABCDEF, vxk12x89ABCDEF);
      const __m128i vp12xGHIJKLMNlo = _mm_mullo_epi16(vxi12xGHIJKLMN, vxk12xGHIJKLMN);
      const __m128i vp12xGHIJKLMNhi = _mm_mulhi_epi16(vxi12xGHIJKLMN, vxk12xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp12x01234567lo, vp12x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp12x01234567lo, vp12x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp12x89ABCDEFlo, vp12x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp12x89ABCDEFlo, vp12x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp12xGHIJKLMNlo, vp12xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp12xGHIJKLMNlo, vp12xGHIJKLMNhi));

      const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
      const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 312 * sizeof(int8_t)));
      const __m128i vi13x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i13 + 8));
      const __m128i vk13x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 320 * sizeof(int8_t)));
      const __m128i vi13xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i13 + 16));
      const __m128i vk13xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 328 * sizeof(int8_t)));
      i13 += 24;

      const __m128i vxi13x01234567 = _mm_unpacklo_epi8(vi13x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi13x01234567));
      const __m128i vxk13x01234567 = _mm_unpacklo_epi8(vk13x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk13x01234567));
      const __m128i vxi13x89ABCDEF = _mm_unpacklo_epi8(vi13x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi13x89ABCDEF));
      const __m128i vxk13x89ABCDEF = _mm_unpacklo_epi8(vk13x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk13x89ABCDEF));
      const __m128i vxi13xGHIJKLMN = _mm_unpacklo_epi8(vi13xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi13xGHIJKLMN));
      const __m128i vxk13xGHIJKLMN = _mm_unpacklo_epi8(vk13xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk13xGHIJKLMN));

      const __m128i vp13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vp13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);
      const __m128i vp13x89ABCDEFlo = _mm_mullo_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF);
      const __m128i vp13x89ABCDEFhi = _mm_mulhi_epi16(vxi13x89ABCDEF, vxk13x89ABCDEF);
      const __m128i vp13xGHIJKLMNlo = _mm_mullo_epi16(vxi13xGHIJKLMN, vxk13xGHIJKLMN);
      const __m128i vp13xGHIJKLMNhi = _mm_mulhi_epi16(vxi13xGHIJKLMN, vxk13xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp13x01234567lo, vp13x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp13x01234567lo, vp13x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp13x89ABCDEFlo, vp13x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp13x89ABCDEFlo, vp13x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp13xGHIJKLMNlo, vp13xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp13xGHIJKLMNlo, vp13xGHIJKLMNhi));

      const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
      const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 336 * sizeof(int8_t)));
      const __m128i vi14x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i14 + 8));
      const __m128i vk14x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 344 * sizeof(int8_t)));
      const __m128i vi14xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i14 + 16));
      const __m128i vk14xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 352 * sizeof(int8_t)));
      i14 += 24;

      const __m128i vxi14x01234567 = _mm_unpacklo_epi8(vi14x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi14x01234567));
      const __m128i vxk14x01234567 = _mm_unpacklo_epi8(vk14x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk14x01234567));
      const __m128i vxi14x89ABCDEF = _mm_unpacklo_epi8(vi14x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi14x89ABCDEF));
      const __m128i vxk14x89ABCDEF = _mm_unpacklo_epi8(vk14x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk14x89ABCDEF));
      const __m128i vxi14xGHIJKLMN = _mm_unpacklo_epi8(vi14xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi14xGHIJKLMN));
      const __m128i vxk14xGHIJKLMN = _mm_unpacklo_epi8(vk14xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk14xGHIJKLMN));

      const __m128i vp14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vp14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);
      const __m128i vp14x89ABCDEFlo = _mm_mullo_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);
      const __m128i vp14x89ABCDEFhi = _mm_mulhi_epi16(vxi14x89ABCDEF, vxk14x89ABCDEF);
      const __m128i vp14xGHIJKLMNlo = _mm_mullo_epi16(vxi14xGHIJKLMN, vxk14xGHIJKLMN);
      const __m128i vp14xGHIJKLMNhi = _mm_mulhi_epi16(vxi14xGHIJKLMN, vxk14xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp14x01234567lo, vp14x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp14x01234567lo, vp14x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp14x89ABCDEFlo, vp14x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp14x89ABCDEFlo, vp14x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp14xGHIJKLMNlo, vp14xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp14xGHIJKLMNlo, vp14xGHIJKLMNhi));

      const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
      const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 360 * sizeof(int8_t)));
      const __m128i vi15x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i15 + 8));
      const __m128i vk15x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 368 * sizeof(int8_t)));
      const __m128i vi15xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i15 + 16));
      const __m128i vk15xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 376 * sizeof(int8_t)));
      i15 += 24;

      const __m128i vxi15x01234567 = _mm_unpacklo_epi8(vi15x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi15x01234567));
      const __m128i vxk15x01234567 = _mm_unpacklo_epi8(vk15x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk15x01234567));
      const __m128i vxi15x89ABCDEF = _mm_unpacklo_epi8(vi15x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi15x89ABCDEF));
      const __m128i vxk15x89ABCDEF = _mm_unpacklo_epi8(vk15x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk15x89ABCDEF));
      const __m128i vxi15xGHIJKLMN = _mm_unpacklo_epi8(vi15xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi15xGHIJKLMN));
      const __m128i vxk15xGHIJKLMN = _mm_unpacklo_epi8(vk15xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk15xGHIJKLMN));

      const __m128i vp15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vp15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);
      const __m128i vp15x89ABCDEFlo = _mm_mullo_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF);
      const __m128i vp15x89ABCDEFhi = _mm_mulhi_epi16(vxi15x89ABCDEF, vxk15x89ABCDEF);
      const __m128i vp15xGHIJKLMNlo = _mm_mullo_epi16(vxi15xGHIJKLMN, vxk15xGHIJKLMN);
      const __m128i vp15xGHIJKLMNhi = _mm_mulhi_epi16(vxi15xGHIJKLMN, vxk15xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp15x01234567lo, vp15x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp15x01234567lo, vp15x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp15x89ABCDEFlo, vp15x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp15x89ABCDEFlo, vp15x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp15xGHIJKLMNlo, vp15xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp15xGHIJKLMNlo, vp15xGHIJKLMNhi));

      const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
      const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 384 * sizeof(int8_t)));
      const __m128i vi16x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i16 + 8));
      const __m128i vk16x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 392 * sizeof(int8_t)));
      const __m128i vi16xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i16 + 16));
      const __m128i vk16xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 400 * sizeof(int8_t)));
      i16 += 24;

      const __m128i vxi16x01234567 = _mm_unpacklo_epi8(vi16x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi16x01234567));
      const __m128i vxk16x01234567 = _mm_unpacklo_epi8(vk16x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk16x01234567));
      const __m128i vxi16x89ABCDEF = _mm_unpacklo_epi8(vi16x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi16x89ABCDEF));
      const __m128i vxk16x89ABCDEF = _mm_unpacklo_epi8(vk16x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk16x89ABCDEF));
      const __m128i vxi16xGHIJKLMN = _mm_unpacklo_epi8(vi16xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi16xGHIJKLMN));
      const __m128i vxk16xGHIJKLMN = _mm_unpacklo_epi8(vk16xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk16xGHIJKLMN));

      const __m128i vp16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vp16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);
      const __m128i vp16x89ABCDEFlo = _mm_mullo_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);
      const __m128i vp16x89ABCDEFhi = _mm_mulhi_epi16(vxi16x89ABCDEF, vxk16x89ABCDEF);
      const __m128i vp16xGHIJKLMNlo = _mm_mullo_epi16(vxi16xGHIJKLMN, vxk16xGHIJKLMN);
      const __m128i vp16xGHIJKLMNhi = _mm_mulhi_epi16(vxi16xGHIJKLMN, vxk16xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp16x01234567lo, vp16x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp16x01234567lo, vp16x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp16x89ABCDEFlo, vp16x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp16x89ABCDEFlo, vp16x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp16xGHIJKLMNlo, vp16xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp16xGHIJKLMNlo, vp16xGHIJKLMNhi));

      const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
      const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 408 * sizeof(int8_t)));
      const __m128i vi17x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i17 + 8));
      const __m128i vk17x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 416 * sizeof(int8_t)));
      const __m128i vi17xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i17 + 16));
      const __m128i vk17xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 424 * sizeof(int8_t)));
      i17 += 24;

      const __m128i vxi17x01234567 = _mm_unpacklo_epi8(vi17x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi17x01234567));
      const __m128i vxk17x01234567 = _mm_unpacklo_epi8(vk17x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk17x01234567));
      const __m128i vxi17x89ABCDEF = _mm_unpacklo_epi8(vi17x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi17x89ABCDEF));
      const __m128i vxk17x89ABCDEF = _mm_unpacklo_epi8(vk17x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk17x89ABCDEF));
      const __m128i vxi17xGHIJKLMN = _mm_unpacklo_epi8(vi17xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi17xGHIJKLMN));
      const __m128i vxk17xGHIJKLMN = _mm_unpacklo_epi8(vk17xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk17xGHIJKLMN));

      const __m128i vp17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vp17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);
      const __m128i vp17x89ABCDEFlo = _mm_mullo_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF);
      const __m128i vp17x89ABCDEFhi = _mm_mulhi_epi16(vxi17x89ABCDEF, vxk17x89ABCDEF);
      const __m128i vp17xGHIJKLMNlo = _mm_mullo_epi16(vxi17xGHIJKLMN, vxk17xGHIJKLMN);
      const __m128i vp17xGHIJKLMNhi = _mm_mulhi_epi16(vxi17xGHIJKLMN, vxk17xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp17x01234567lo, vp17x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp17x01234567lo, vp17x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp17x89ABCDEFlo, vp17x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp17x89ABCDEFlo, vp17x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp17xGHIJKLMNlo, vp17xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp17xGHIJKLMNlo, vp17xGHIJKLMNhi));

      const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
      const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 432 * sizeof(int8_t)));
      const __m128i vi18x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i18 + 8));
      const __m128i vk18x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 440 * sizeof(int8_t)));
      const __m128i vi18xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i18 + 16));
      const __m128i vk18xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 448 * sizeof(int8_t)));
      i18 += 24;

      const __m128i vxi18x01234567 = _mm_unpacklo_epi8(vi18x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi18x01234567));
      const __m128i vxk18x01234567 = _mm_unpacklo_epi8(vk18x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk18x01234567));
      const __m128i vxi18x89ABCDEF = _mm_unpacklo_epi8(vi18x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi18x89ABCDEF));
      const __m128i vxk18x89ABCDEF = _mm_unpacklo_epi8(vk18x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk18x89ABCDEF));
      const __m128i vxi18xGHIJKLMN = _mm_unpacklo_epi8(vi18xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi18xGHIJKLMN));
      const __m128i vxk18xGHIJKLMN = _mm_unpacklo_epi8(vk18xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk18xGHIJKLMN));

      const __m128i vp18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vp18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);
      const __m128i vp18x89ABCDEFlo = _mm_mullo_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);
      const __m128i vp18x89ABCDEFhi = _mm_mulhi_epi16(vxi18x89ABCDEF, vxk18x89ABCDEF);
      const __m128i vp18xGHIJKLMNlo = _mm_mullo_epi16(vxi18xGHIJKLMN, vxk18xGHIJKLMN);
      const __m128i vp18xGHIJKLMNhi = _mm_mulhi_epi16(vxi18xGHIJKLMN, vxk18xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp18x01234567lo, vp18x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp18x01234567lo, vp18x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp18x89ABCDEFlo, vp18x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp18x89ABCDEFlo, vp18x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp18xGHIJKLMNlo, vp18xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp18xGHIJKLMNlo, vp18xGHIJKLMNhi));

      const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
      const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 456 * sizeof(int8_t)));
      const __m128i vi19x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i19 + 8));
      const __m128i vk19x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 464 * sizeof(int8_t)));
      const __m128i vi19xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i19 + 16));
      const __m128i vk19xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 472 * sizeof(int8_t)));
      i19 += 24;

      const __m128i vxi19x01234567 = _mm_unpacklo_epi8(vi19x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi19x01234567));
      const __m128i vxk19x01234567 = _mm_unpacklo_epi8(vk19x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk19x01234567));
      const __m128i vxi19x89ABCDEF = _mm_unpacklo_epi8(vi19x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi19x89ABCDEF));
      const __m128i vxk19x89ABCDEF = _mm_unpacklo_epi8(vk19x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk19x89ABCDEF));
      const __m128i vxi19xGHIJKLMN = _mm_unpacklo_epi8(vi19xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi19xGHIJKLMN));
      const __m128i vxk19xGHIJKLMN = _mm_unpacklo_epi8(vk19xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk19xGHIJKLMN));

      const __m128i vp19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vp19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);
      const __m128i vp19x89ABCDEFlo = _mm_mullo_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF);
      const __m128i vp19x89ABCDEFhi = _mm_mulhi_epi16(vxi19x89ABCDEF, vxk19x89ABCDEF);
      const __m128i vp19xGHIJKLMNlo = _mm_mullo_epi16(vxi19xGHIJKLMN, vxk19xGHIJKLMN);
      const __m128i vp19xGHIJKLMNhi = _mm_mulhi_epi16(vxi19xGHIJKLMN, vxk19xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp19x01234567lo, vp19x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp19x01234567lo, vp19x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp19x89ABCDEFlo, vp19x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp19x89ABCDEFlo, vp19x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp19xGHIJKLMNlo, vp19xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp19xGHIJKLMNlo, vp19xGHIJKLMNhi));

      const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
      const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 480 * sizeof(int8_t)));
      const __m128i vi20x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i20 + 8));
      const __m128i vk20x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 488 * sizeof(int8_t)));
      const __m128i vi20xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i20 + 16));
      const __m128i vk20xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 496 * sizeof(int8_t)));
      i20 += 24;

      const __m128i vxi20x01234567 = _mm_unpacklo_epi8(vi20x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi20x01234567));
      const __m128i vxk20x01234567 = _mm_unpacklo_epi8(vk20x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk20x01234567));
      const __m128i vxi20x89ABCDEF = _mm_unpacklo_epi8(vi20x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi20x89ABCDEF));
      const __m128i vxk20x89ABCDEF = _mm_unpacklo_epi8(vk20x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk20x89ABCDEF));
      const __m128i vxi20xGHIJKLMN = _mm_unpacklo_epi8(vi20xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi20xGHIJKLMN));
      const __m128i vxk20xGHIJKLMN = _mm_unpacklo_epi8(vk20xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk20xGHIJKLMN));

      const __m128i vp20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vp20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);
      const __m128i vp20x89ABCDEFlo = _mm_mullo_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);
      const __m128i vp20x89ABCDEFhi = _mm_mulhi_epi16(vxi20x89ABCDEF, vxk20x89ABCDEF);
      const __m128i vp20xGHIJKLMNlo = _mm_mullo_epi16(vxi20xGHIJKLMN, vxk20xGHIJKLMN);
      const __m128i vp20xGHIJKLMNhi = _mm_mulhi_epi16(vxi20xGHIJKLMN, vxk20xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp20x01234567lo, vp20x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp20x01234567lo, vp20x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp20x89ABCDEFlo, vp20x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp20x89ABCDEFlo, vp20x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp20xGHIJKLMNlo, vp20xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp20xGHIJKLMNlo, vp20xGHIJKLMNhi));

      const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
      const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 504 * sizeof(int8_t)));
      const __m128i vi21x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i21 + 8));
      const __m128i vk21x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 512 * sizeof(int8_t)));
      const __m128i vi21xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i21 + 16));
      const __m128i vk21xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 520 * sizeof(int8_t)));
      i21 += 24;

      const __m128i vxi21x01234567 = _mm_unpacklo_epi8(vi21x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi21x01234567));
      const __m128i vxk21x01234567 = _mm_unpacklo_epi8(vk21x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk21x01234567));
      const __m128i vxi21x89ABCDEF = _mm_unpacklo_epi8(vi21x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi21x89ABCDEF));
      const __m128i vxk21x89ABCDEF = _mm_unpacklo_epi8(vk21x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk21x89ABCDEF));
      const __m128i vxi21xGHIJKLMN = _mm_unpacklo_epi8(vi21xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi21xGHIJKLMN));
      const __m128i vxk21xGHIJKLMN = _mm_unpacklo_epi8(vk21xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk21xGHIJKLMN));

      const __m128i vp21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vp21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);
      const __m128i vp21x89ABCDEFlo = _mm_mullo_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF);
      const __m128i vp21x89ABCDEFhi = _mm_mulhi_epi16(vxi21x89ABCDEF, vxk21x89ABCDEF);
      const __m128i vp21xGHIJKLMNlo = _mm_mullo_epi16(vxi21xGHIJKLMN, vxk21xGHIJKLMN);
      const __m128i vp21xGHIJKLMNhi = _mm_mulhi_epi16(vxi21xGHIJKLMN, vxk21xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp21x01234567lo, vp21x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp21x01234567lo, vp21x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp21x89ABCDEFlo, vp21x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp21x89ABCDEFlo, vp21x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp21xGHIJKLMNlo, vp21xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp21xGHIJKLMNlo, vp21xGHIJKLMNhi));

      const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
      const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 528 * sizeof(int8_t)));
      const __m128i vi22x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i22 + 8));
      const __m128i vk22x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 536 * sizeof(int8_t)));
      const __m128i vi22xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i22 + 16));
      const __m128i vk22xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 544 * sizeof(int8_t)));
      i22 += 24;

      const __m128i vxi22x01234567 = _mm_unpacklo_epi8(vi22x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi22x01234567));
      const __m128i vxk22x01234567 = _mm_unpacklo_epi8(vk22x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk22x01234567));
      const __m128i vxi22x89ABCDEF = _mm_unpacklo_epi8(vi22x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi22x89ABCDEF));
      const __m128i vxk22x89ABCDEF = _mm_unpacklo_epi8(vk22x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk22x89ABCDEF));
      const __m128i vxi22xGHIJKLMN = _mm_unpacklo_epi8(vi22xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi22xGHIJKLMN));
      const __m128i vxk22xGHIJKLMN = _mm_unpacklo_epi8(vk22xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk22xGHIJKLMN));

      const __m128i vp22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vp22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);
      const __m128i vp22x89ABCDEFlo = _mm_mullo_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);
      const __m128i vp22x89ABCDEFhi = _mm_mulhi_epi16(vxi22x89ABCDEF, vxk22x89ABCDEF);
      const __m128i vp22xGHIJKLMNlo = _mm_mullo_epi16(vxi22xGHIJKLMN, vxk22xGHIJKLMN);
      const __m128i vp22xGHIJKLMNhi = _mm_mulhi_epi16(vxi22xGHIJKLMN, vxk22xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp22x01234567lo, vp22x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp22x01234567lo, vp22x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp22x89ABCDEFlo, vp22x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp22x89ABCDEFlo, vp22x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp22xGHIJKLMNlo, vp22xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp22xGHIJKLMNlo, vp22xGHIJKLMNhi));

      const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
      const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 552 * sizeof(int8_t)));
      const __m128i vi23x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i23 + 8));
      const __m128i vk23x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 560 * sizeof(int8_t)));
      const __m128i vi23xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i23 + 16));
      const __m128i vk23xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 568 * sizeof(int8_t)));
      i23 += 24;

      const __m128i vxi23x01234567 = _mm_unpacklo_epi8(vi23x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi23x01234567));
      const __m128i vxk23x01234567 = _mm_unpacklo_epi8(vk23x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk23x01234567));
      const __m128i vxi23x89ABCDEF = _mm_unpacklo_epi8(vi23x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi23x89ABCDEF));
      const __m128i vxk23x89ABCDEF = _mm_unpacklo_epi8(vk23x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk23x89ABCDEF));
      const __m128i vxi23xGHIJKLMN = _mm_unpacklo_epi8(vi23xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi23xGHIJKLMN));
      const __m128i vxk23xGHIJKLMN = _mm_unpacklo_epi8(vk23xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk23xGHIJKLMN));

      const __m128i vp23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vp23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);
      const __m128i vp23x89ABCDEFlo = _mm_mullo_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF);
      const __m128i vp23x89ABCDEFhi = _mm_mulhi_epi16(vxi23x89ABCDEF, vxk23x89ABCDEF);
      const __m128i vp23xGHIJKLMNlo = _mm_mullo_epi16(vxi23xGHIJKLMN, vxk23xGHIJKLMN);
      const __m128i vp23xGHIJKLMNhi = _mm_mulhi_epi16(vxi23xGHIJKLMN, vxk23xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp23x01234567lo, vp23x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp23x01234567lo, vp23x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp23x89ABCDEFlo, vp23x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp23x89ABCDEFlo, vp23x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp23xGHIJKLMNlo, vp23xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp23xGHIJKLMNlo, vp23xGHIJKLMNhi));

      const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
      const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 576 * sizeof(int8_t)));
      const __m128i vi24x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i24 + 8));
      const __m128i vk24x89ABCDEF = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 584 * sizeof(int8_t)));
      const __m128i vi24xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i24 + 16));
      const __m128i vk24xGHIJKLMN = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24 * sizeof(int32_t) + 592 * sizeof(int8_t)));
      i24 += 24;

      const __m128i vxi24x01234567 = _mm_unpacklo_epi8(vi24x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi24x01234567));
      const __m128i vxk24x01234567 = _mm_unpacklo_epi8(vk24x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk24x01234567));
      const __m128i vxi24x89ABCDEF = _mm_unpacklo_epi8(vi24x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi24x89ABCDEF));
      const __m128i vxk24x89ABCDEF = _mm_unpacklo_epi8(vk24x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vk24x89ABCDEF));
      const __m128i vxi24xGHIJKLMN = _mm_unpacklo_epi8(vi24xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi24xGHIJKLMN));
      const __m128i vxk24xGHIJKLMN = _mm_unpacklo_epi8(vk24xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vk24xGHIJKLMN));

      const __m128i vp24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vp24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);
      const __m128i vp24x89ABCDEFlo = _mm_mullo_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);
      const __m128i vp24x89ABCDEFhi = _mm_mulhi_epi16(vxi24x89ABCDEF, vxk24x89ABCDEF);
      const __m128i vp24xGHIJKLMNlo = _mm_mullo_epi16(vxi24xGHIJKLMN, vxk24xGHIJKLMN);
      const __m128i vp24xGHIJKLMNhi = _mm_mulhi_epi16(vxi24xGHIJKLMN, vxk24xGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp24x01234567lo, vp24x01234567hi));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp24x01234567lo, vp24x01234567hi));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_unpacklo_epi16(vp24x89ABCDEFlo, vp24x89ABCDEFhi));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_unpackhi_epi16(vp24x89ABCDEFlo, vp24x89ABCDEFhi));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_unpacklo_epi16(vp24xGHIJKLMNlo, vp24xGHIJKLMNhi));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_unpackhi_epi16(vp24xGHIJKLMNlo, vp24xGHIJKLMNhi));

      w = (const void*) ((uintptr_t) w + 24 * sizeof(int32_t) + 600 * sizeof(int8_t));

      const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.multiplier);
      const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.rounding);

      const __m128i vnmask0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123);
      const __m128i vnmask4567 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567);
      const __m128i vnmask89AB = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc89AB);
      const __m128i vnmaskCDEF = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccCDEF);
      const __m128i vnmaskGHIJ = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccGHIJ);
      const __m128i vnmaskKLMN = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccKLMN);

      const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
      const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);
      const __m128i vabsacc89AB = _mm_abs_epi32(vacc89AB);
      const __m128i vabsaccCDEF = _mm_abs_epi32(vaccCDEF);
      const __m128i vabsaccGHIJ = _mm_abs_epi32(vaccGHIJ);
      const __m128i vabsaccKLMN = _mm_abs_epi32(vaccKLMN);

      const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
      const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
      const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
      const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);
      const __m128i vabsacc9B = _mm_shuffle_epi32(vabsacc89AB, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprod8A = _mm_mul_epu32(vabsacc89AB, vmultiplier);
      const __m128i vabsprod9B = _mm_mul_epu32(vabsacc9B, vmultiplier);
      const __m128i vabsaccDF = _mm_shuffle_epi32(vabsaccCDEF, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprodCE = _mm_mul_epu32(vabsaccCDEF, vmultiplier);
      const __m128i vabsprodDF = _mm_mul_epu32(vabsaccDF, vmultiplier);
      const __m128i vabsaccHJ = _mm_shuffle_epi32(vabsaccGHIJ, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprodGI = _mm_mul_epu32(vabsaccGHIJ, vmultiplier);
      const __m128i vabsprodHJ = _mm_mul_epu32(vabsaccHJ, vmultiplier);
      const __m128i vabsaccLN = _mm_shuffle_epi32(vabsaccKLMN, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsprodKM = _mm_mul_epu32(vabsaccKLMN, vmultiplier);
      const __m128i vabsprodLN = _mm_mul_epu32(vabsaccLN, vmultiplier);

      const __m128i vnmask02 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask13 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmask46 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask57 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmask8A = _mm_shuffle_epi32(vnmask89AB, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask9B = _mm_shuffle_epi32(vnmask89AB, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmaskCE = _mm_shuffle_epi32(vnmaskCDEF, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmaskDF = _mm_shuffle_epi32(vnmaskCDEF, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmaskGI = _mm_shuffle_epi32(vnmaskGHIJ, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmaskHJ = _mm_shuffle_epi32(vnmaskGHIJ, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmaskKM = _mm_shuffle_epi32(vnmaskKLMN, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmaskLN = _mm_shuffle_epi32(vnmaskKLMN, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vprod02 = _mm_sub_epi64(_mm_xor_si128(vabsprod02, vnmask02), vnmask02);
      const __m128i vprod13 = _mm_sub_epi64(_mm_xor_si128(vabsprod13, vnmask13), vnmask13);
      const __m128i vprod46 = _mm_sub_epi64(_mm_xor_si128(vabsprod46, vnmask46), vnmask46);
      const __m128i vprod57 = _mm_sub_epi64(_mm_xor_si128(vabsprod57, vnmask57), vnmask57);
      const __m128i vprod8A = _mm_sub_epi64(_mm_xor_si128(vabsprod8A, vnmask8A), vnmask8A);
      const __m128i vprod9B = _mm_sub_epi64(_mm_xor_si128(vabsprod9B, vnmask9B), vnmask9B);
      const __m128i vprodCE = _mm_sub_epi64(_mm_xor_si128(vabsprodCE, vnmaskCE), vnmaskCE);
      const __m128i vprodDF = _mm_sub_epi64(_mm_xor_si128(vabsprodDF, vnmaskDF), vnmaskDF);
      const __m128i vprodGI = _mm_sub_epi64(_mm_xor_si128(vabsprodGI, vnmaskGI), vnmaskGI);
      const __m128i vprodHJ = _mm_sub_epi64(_mm_xor_si128(vabsprodHJ, vnmaskHJ), vnmaskHJ);
      const __m128i vprodKM = _mm_sub_epi64(_mm_xor_si128(vabsprodKM, vnmaskKM), vnmaskKM);
      const __m128i vprodLN = _mm_sub_epi64(_mm_xor_si128(vabsprodLN, vnmaskLN), vnmaskLN);

      const __m128i vq31prod02 = _mm_srli_epi64(_mm_add_epi64(vprod02, vrounding), 31);
      const __m128i vq31prod13 = _mm_srli_epi64(_mm_add_epi64(vprod13, vrounding), 31);
      const __m128i vq31prod46 = _mm_srli_epi64(_mm_add_epi64(vprod46, vrounding), 31);
      const __m128i vq31prod57 = _mm_srli_epi64(_mm_add_epi64(vprod57, vrounding), 31);
      const __m128i vq31prod8A = _mm_srli_epi64(_mm_add_epi64(vprod8A, vrounding), 31);
      const __m128i vq31prod9B = _mm_srli_epi64(_mm_add_epi64(vprod9B, vrounding), 31);
      const __m128i vq31prodCE = _mm_srli_epi64(_mm_add_epi64(vprodCE, vrounding), 31);
      const __m128i vq31prodDF = _mm_srli_epi64(_mm_add_epi64(vprodDF, vrounding), 31);
      const __m128i vq31prodGI = _mm_srli_epi64(_mm_add_epi64(vprodGI, vrounding), 31);
      const __m128i vq31prodHJ = _mm_srli_epi64(_mm_add_epi64(vprodHJ, vrounding), 31);
      const __m128i vq31prodKM = _mm_srli_epi64(_mm_add_epi64(vprodKM, vrounding), 31);
      const __m128i vq31prodLN = _mm_srli_epi64(_mm_add_epi64(vprodLN, vrounding), 31);

      const __m128i vq31prod0213 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod02), _mm_castsi128_ps(vq31prod13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prod4657 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod46), _mm_castsi128_ps(vq31prod57), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prod8A9B = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod8A), _mm_castsi128_ps(vq31prod9B), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prodCEDF = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prodCE), _mm_castsi128_ps(vq31prodDF), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prodGIHJ = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prodGI), _mm_castsi128_ps(vq31prodHJ), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prodKMLN = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prodKM), _mm_castsi128_ps(vq31prodLN), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vq31prod0123 = _mm_shuffle_epi32(vq31prod0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prod4567 = _mm_shuffle_epi32(vq31prod4657, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prod89AB = _mm_shuffle_epi32(vq31prod8A9B, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prodCDEF = _mm_shuffle_epi32(vq31prodCEDF, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prodGHIJ = _mm_shuffle_epi32(vq31prodGIHJ, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prodKLMN = _mm_shuffle_epi32(vq31prodKMLN, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_mask);
      const __m128i vrem0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));
      const __m128i vrem4567 =
        _mm_add_epi32(_mm_and_si128(vq31prod4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod4567));
      const __m128i vrem89AB =
        _mm_add_epi32(_mm_and_si128(vq31prod89AB, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod89AB));
      const __m128i vremCDEF =
        _mm_add_epi32(_mm_and_si128(vq31prodCDEF, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prodCDEF));
      const __m128i vremGHIJ =
        _mm_add_epi32(_mm_and_si128(vq31prodGHIJ, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prodGHIJ));
      const __m128i vremKLMN =
        _mm_add_epi32(_mm_and_si128(vq31prodKLMN, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prodKLMN));

      const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_threshold);
      const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse2.shift);
      vacc0123 =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
      vacc4567 =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));
      vacc89AB =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod89AB, vshift), _mm_cmpgt_epi32(vrem89AB, vremainder_threshold));
      vaccCDEF =
        _mm_sub_epi32(_mm_sra_epi32(vq31prodCDEF, vshift), _mm_cmpgt_epi32(vremCDEF, vremainder_threshold));
      vaccGHIJ =
        _mm_sub_epi32(_mm_sra_epi32(vq31prodGHIJ, vshift), _mm_cmpgt_epi32(vremGHIJ, vremainder_threshold));
      vaccKLMN =
        _mm_sub_epi32(_mm_sra_epi32(vq31prodKLMN, vshift), _mm_cmpgt_epi32(vremKLMN, vremainder_threshold));

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);
      __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(vaccGHIJ, vaccKLMN), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_min);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
      vout89ABCDEF = _mm_max_epi16(vout89ABCDEF, voutput_min);
      voutGHIJKLMN = _mm_max_epi16(voutGHIJKLMN, voutput_min);

      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_max);
      vout01234567 = _mm_min_epi16(vout01234567, voutput_max);
      vout89ABCDEF = _mm_min_epi16(vout89ABCDEF, voutput_max);
      voutGHIJKLMN = _mm_min_epi16(voutGHIJKLMN, voutput_max);

      __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
      __m128i voutGHIJKLMNGHIJKLMN = _mm_packs_epi16(voutGHIJKLMN, voutGHIJKLMN);


      _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
      _mm_storel_epi64((__m128i*) (output + 16), voutGHIJKLMNGHIJKLMN);
      output += 24;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 24 * sizeof(int32_t));
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
        __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 4 * sizeof(int32_t)));


        const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vk0x01234567 = _mm_loadl_epi64((const __m128i*) k);
        i0 += 8;

        const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
        const __m128i vxk0x01234567 = _mm_unpacklo_epi8(vk0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk0x01234567));

        const __m128i vp0x01234567lo = _mm_mullo_epi16(vxi0x01234567, vxk0x01234567);
        const __m128i vp0x01234567hi = _mm_mulhi_epi16(vxi0x01234567, vxk0x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp0x01234567lo, vp0x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp0x01234567lo, vp0x01234567hi));

        const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vk1x01234567 = _mm_loadl_epi64((const __m128i*) (k + 24));
        i1 += 8;

        const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
        const __m128i vxk1x01234567 = _mm_unpacklo_epi8(vk1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk1x01234567));

        const __m128i vp1x01234567lo = _mm_mullo_epi16(vxi1x01234567, vxk1x01234567);
        const __m128i vp1x01234567hi = _mm_mulhi_epi16(vxi1x01234567, vxk1x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp1x01234567lo, vp1x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp1x01234567lo, vp1x01234567hi));

        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vk2x01234567 = _mm_loadl_epi64((const __m128i*) (k + 48));
        i2 += 8;

        const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
        const __m128i vxk2x01234567 = _mm_unpacklo_epi8(vk2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk2x01234567));

        const __m128i vp2x01234567lo = _mm_mullo_epi16(vxi2x01234567, vxk2x01234567);
        const __m128i vp2x01234567hi = _mm_mulhi_epi16(vxi2x01234567, vxk2x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp2x01234567lo, vp2x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp2x01234567lo, vp2x01234567hi));

        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vk3x01234567 = _mm_loadl_epi64((const __m128i*) (k + 72));
        i3 += 8;

        const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
        const __m128i vxk3x01234567 = _mm_unpacklo_epi8(vk3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk3x01234567));

        const __m128i vp3x01234567lo = _mm_mullo_epi16(vxi3x01234567, vxk3x01234567);
        const __m128i vp3x01234567hi = _mm_mulhi_epi16(vxi3x01234567, vxk3x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp3x01234567lo, vp3x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp3x01234567lo, vp3x01234567hi));

        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vk4x01234567 = _mm_loadl_epi64((const __m128i*) (k + 96));
        i4 += 8;

        const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
        const __m128i vxk4x01234567 = _mm_unpacklo_epi8(vk4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk4x01234567));

        const __m128i vp4x01234567lo = _mm_mullo_epi16(vxi4x01234567, vxk4x01234567);
        const __m128i vp4x01234567hi = _mm_mulhi_epi16(vxi4x01234567, vxk4x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp4x01234567lo, vp4x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp4x01234567lo, vp4x01234567hi));

        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vk5x01234567 = _mm_loadl_epi64((const __m128i*) (k + 120));
        i5 += 8;

        const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
        const __m128i vxk5x01234567 = _mm_unpacklo_epi8(vk5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk5x01234567));

        const __m128i vp5x01234567lo = _mm_mullo_epi16(vxi5x01234567, vxk5x01234567);
        const __m128i vp5x01234567hi = _mm_mulhi_epi16(vxi5x01234567, vxk5x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp5x01234567lo, vp5x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp5x01234567lo, vp5x01234567hi));

        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vk6x01234567 = _mm_loadl_epi64((const __m128i*) (k + 144));
        i6 += 8;

        const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));
        const __m128i vxk6x01234567 = _mm_unpacklo_epi8(vk6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk6x01234567));

        const __m128i vp6x01234567lo = _mm_mullo_epi16(vxi6x01234567, vxk6x01234567);
        const __m128i vp6x01234567hi = _mm_mulhi_epi16(vxi6x01234567, vxk6x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp6x01234567lo, vp6x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp6x01234567lo, vp6x01234567hi));

        const __m128i vi7x01234567 = _mm_loadl_epi64((const __m128i*) i7);
        const __m128i vk7x01234567 = _mm_loadl_epi64((const __m128i*) (k + 168));
        i7 += 8;

        const __m128i vxi7x01234567 = _mm_unpacklo_epi8(vi7x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi7x01234567));
        const __m128i vxk7x01234567 = _mm_unpacklo_epi8(vk7x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk7x01234567));

        const __m128i vp7x01234567lo = _mm_mullo_epi16(vxi7x01234567, vxk7x01234567);
        const __m128i vp7x01234567hi = _mm_mulhi_epi16(vxi7x01234567, vxk7x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp7x01234567lo, vp7x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp7x01234567lo, vp7x01234567hi));

        const __m128i vi8x01234567 = _mm_loadl_epi64((const __m128i*) i8);
        const __m128i vk8x01234567 = _mm_loadl_epi64((const __m128i*) (k + 192));
        i8 += 8;

        const __m128i vxi8x01234567 = _mm_unpacklo_epi8(vi8x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi8x01234567));
        const __m128i vxk8x01234567 = _mm_unpacklo_epi8(vk8x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk8x01234567));

        const __m128i vp8x01234567lo = _mm_mullo_epi16(vxi8x01234567, vxk8x01234567);
        const __m128i vp8x01234567hi = _mm_mulhi_epi16(vxi8x01234567, vxk8x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp8x01234567lo, vp8x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp8x01234567lo, vp8x01234567hi));

        const __m128i vi9x01234567 = _mm_loadl_epi64((const __m128i*) i9);
        const __m128i vk9x01234567 = _mm_loadl_epi64((const __m128i*) (k + 216));
        i9 += 8;

        const __m128i vxi9x01234567 = _mm_unpacklo_epi8(vi9x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi9x01234567));
        const __m128i vxk9x01234567 = _mm_unpacklo_epi8(vk9x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk9x01234567));

        const __m128i vp9x01234567lo = _mm_mullo_epi16(vxi9x01234567, vxk9x01234567);
        const __m128i vp9x01234567hi = _mm_mulhi_epi16(vxi9x01234567, vxk9x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp9x01234567lo, vp9x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp9x01234567lo, vp9x01234567hi));

        const __m128i vi10x01234567 = _mm_loadl_epi64((const __m128i*) i10);
        const __m128i vk10x01234567 = _mm_loadl_epi64((const __m128i*) (k + 240));
        i10 += 8;

        const __m128i vxi10x01234567 = _mm_unpacklo_epi8(vi10x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi10x01234567));
        const __m128i vxk10x01234567 = _mm_unpacklo_epi8(vk10x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk10x01234567));

        const __m128i vp10x01234567lo = _mm_mullo_epi16(vxi10x01234567, vxk10x01234567);
        const __m128i vp10x01234567hi = _mm_mulhi_epi16(vxi10x01234567, vxk10x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp10x01234567lo, vp10x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp10x01234567lo, vp10x01234567hi));

        const __m128i vi11x01234567 = _mm_loadl_epi64((const __m128i*) i11);
        const __m128i vk11x01234567 = _mm_loadl_epi64((const __m128i*) (k + 264));
        i11 += 8;

        const __m128i vxi11x01234567 = _mm_unpacklo_epi8(vi11x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi11x01234567));
        const __m128i vxk11x01234567 = _mm_unpacklo_epi8(vk11x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk11x01234567));

        const __m128i vp11x01234567lo = _mm_mullo_epi16(vxi11x01234567, vxk11x01234567);
        const __m128i vp11x01234567hi = _mm_mulhi_epi16(vxi11x01234567, vxk11x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp11x01234567lo, vp11x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp11x01234567lo, vp11x01234567hi));

        const __m128i vi12x01234567 = _mm_loadl_epi64((const __m128i*) i12);
        const __m128i vk12x01234567 = _mm_loadl_epi64((const __m128i*) (k + 288));
        i12 += 8;

        const __m128i vxi12x01234567 = _mm_unpacklo_epi8(vi12x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi12x01234567));
        const __m128i vxk12x01234567 = _mm_unpacklo_epi8(vk12x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk12x01234567));

        const __m128i vp12x01234567lo = _mm_mullo_epi16(vxi12x01234567, vxk12x01234567);
        const __m128i vp12x01234567hi = _mm_mulhi_epi16(vxi12x01234567, vxk12x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp12x01234567lo, vp12x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp12x01234567lo, vp12x01234567hi));

        const __m128i vi13x01234567 = _mm_loadl_epi64((const __m128i*) i13);
        const __m128i vk13x01234567 = _mm_loadl_epi64((const __m128i*) (k + 312));
        i13 += 8;

        const __m128i vxi13x01234567 = _mm_unpacklo_epi8(vi13x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi13x01234567));
        const __m128i vxk13x01234567 = _mm_unpacklo_epi8(vk13x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk13x01234567));

        const __m128i vp13x01234567lo = _mm_mullo_epi16(vxi13x01234567, vxk13x01234567);
        const __m128i vp13x01234567hi = _mm_mulhi_epi16(vxi13x01234567, vxk13x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp13x01234567lo, vp13x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp13x01234567lo, vp13x01234567hi));

        const __m128i vi14x01234567 = _mm_loadl_epi64((const __m128i*) i14);
        const __m128i vk14x01234567 = _mm_loadl_epi64((const __m128i*) (k + 336));
        i14 += 8;

        const __m128i vxi14x01234567 = _mm_unpacklo_epi8(vi14x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi14x01234567));
        const __m128i vxk14x01234567 = _mm_unpacklo_epi8(vk14x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk14x01234567));

        const __m128i vp14x01234567lo = _mm_mullo_epi16(vxi14x01234567, vxk14x01234567);
        const __m128i vp14x01234567hi = _mm_mulhi_epi16(vxi14x01234567, vxk14x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp14x01234567lo, vp14x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp14x01234567lo, vp14x01234567hi));

        const __m128i vi15x01234567 = _mm_loadl_epi64((const __m128i*) i15);
        const __m128i vk15x01234567 = _mm_loadl_epi64((const __m128i*) (k + 360));
        i15 += 8;

        const __m128i vxi15x01234567 = _mm_unpacklo_epi8(vi15x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi15x01234567));
        const __m128i vxk15x01234567 = _mm_unpacklo_epi8(vk15x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk15x01234567));

        const __m128i vp15x01234567lo = _mm_mullo_epi16(vxi15x01234567, vxk15x01234567);
        const __m128i vp15x01234567hi = _mm_mulhi_epi16(vxi15x01234567, vxk15x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp15x01234567lo, vp15x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp15x01234567lo, vp15x01234567hi));

        const __m128i vi16x01234567 = _mm_loadl_epi64((const __m128i*) i16);
        const __m128i vk16x01234567 = _mm_loadl_epi64((const __m128i*) (k + 384));
        i16 += 8;

        const __m128i vxi16x01234567 = _mm_unpacklo_epi8(vi16x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi16x01234567));
        const __m128i vxk16x01234567 = _mm_unpacklo_epi8(vk16x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk16x01234567));

        const __m128i vp16x01234567lo = _mm_mullo_epi16(vxi16x01234567, vxk16x01234567);
        const __m128i vp16x01234567hi = _mm_mulhi_epi16(vxi16x01234567, vxk16x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp16x01234567lo, vp16x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp16x01234567lo, vp16x01234567hi));

        const __m128i vi17x01234567 = _mm_loadl_epi64((const __m128i*) i17);
        const __m128i vk17x01234567 = _mm_loadl_epi64((const __m128i*) (k + 408));
        i17 += 8;

        const __m128i vxi17x01234567 = _mm_unpacklo_epi8(vi17x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi17x01234567));
        const __m128i vxk17x01234567 = _mm_unpacklo_epi8(vk17x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk17x01234567));

        const __m128i vp17x01234567lo = _mm_mullo_epi16(vxi17x01234567, vxk17x01234567);
        const __m128i vp17x01234567hi = _mm_mulhi_epi16(vxi17x01234567, vxk17x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp17x01234567lo, vp17x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp17x01234567lo, vp17x01234567hi));

        const __m128i vi18x01234567 = _mm_loadl_epi64((const __m128i*) i18);
        const __m128i vk18x01234567 = _mm_loadl_epi64((const __m128i*) (k + 432));
        i18 += 8;

        const __m128i vxi18x01234567 = _mm_unpacklo_epi8(vi18x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi18x01234567));
        const __m128i vxk18x01234567 = _mm_unpacklo_epi8(vk18x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk18x01234567));

        const __m128i vp18x01234567lo = _mm_mullo_epi16(vxi18x01234567, vxk18x01234567);
        const __m128i vp18x01234567hi = _mm_mulhi_epi16(vxi18x01234567, vxk18x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp18x01234567lo, vp18x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp18x01234567lo, vp18x01234567hi));

        const __m128i vi19x01234567 = _mm_loadl_epi64((const __m128i*) i19);
        const __m128i vk19x01234567 = _mm_loadl_epi64((const __m128i*) (k + 456));
        i19 += 8;

        const __m128i vxi19x01234567 = _mm_unpacklo_epi8(vi19x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi19x01234567));
        const __m128i vxk19x01234567 = _mm_unpacklo_epi8(vk19x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk19x01234567));

        const __m128i vp19x01234567lo = _mm_mullo_epi16(vxi19x01234567, vxk19x01234567);
        const __m128i vp19x01234567hi = _mm_mulhi_epi16(vxi19x01234567, vxk19x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp19x01234567lo, vp19x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp19x01234567lo, vp19x01234567hi));

        const __m128i vi20x01234567 = _mm_loadl_epi64((const __m128i*) i20);
        const __m128i vk20x01234567 = _mm_loadl_epi64((const __m128i*) (k + 480));
        i20 += 8;

        const __m128i vxi20x01234567 = _mm_unpacklo_epi8(vi20x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi20x01234567));
        const __m128i vxk20x01234567 = _mm_unpacklo_epi8(vk20x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk20x01234567));

        const __m128i vp20x01234567lo = _mm_mullo_epi16(vxi20x01234567, vxk20x01234567);
        const __m128i vp20x01234567hi = _mm_mulhi_epi16(vxi20x01234567, vxk20x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp20x01234567lo, vp20x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp20x01234567lo, vp20x01234567hi));

        const __m128i vi21x01234567 = _mm_loadl_epi64((const __m128i*) i21);
        const __m128i vk21x01234567 = _mm_loadl_epi64((const __m128i*) (k + 504));
        i21 += 8;

        const __m128i vxi21x01234567 = _mm_unpacklo_epi8(vi21x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi21x01234567));
        const __m128i vxk21x01234567 = _mm_unpacklo_epi8(vk21x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk21x01234567));

        const __m128i vp21x01234567lo = _mm_mullo_epi16(vxi21x01234567, vxk21x01234567);
        const __m128i vp21x01234567hi = _mm_mulhi_epi16(vxi21x01234567, vxk21x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp21x01234567lo, vp21x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp21x01234567lo, vp21x01234567hi));

        const __m128i vi22x01234567 = _mm_loadl_epi64((const __m128i*) i22);
        const __m128i vk22x01234567 = _mm_loadl_epi64((const __m128i*) (k + 528));
        i22 += 8;

        const __m128i vxi22x01234567 = _mm_unpacklo_epi8(vi22x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi22x01234567));
        const __m128i vxk22x01234567 = _mm_unpacklo_epi8(vk22x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk22x01234567));

        const __m128i vp22x01234567lo = _mm_mullo_epi16(vxi22x01234567, vxk22x01234567);
        const __m128i vp22x01234567hi = _mm_mulhi_epi16(vxi22x01234567, vxk22x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp22x01234567lo, vp22x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp22x01234567lo, vp22x01234567hi));

        const __m128i vi23x01234567 = _mm_loadl_epi64((const __m128i*) i23);
        const __m128i vk23x01234567 = _mm_loadl_epi64((const __m128i*) (k + 552));
        i23 += 8;

        const __m128i vxi23x01234567 = _mm_unpacklo_epi8(vi23x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi23x01234567));
        const __m128i vxk23x01234567 = _mm_unpacklo_epi8(vk23x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk23x01234567));

        const __m128i vp23x01234567lo = _mm_mullo_epi16(vxi23x01234567, vxk23x01234567);
        const __m128i vp23x01234567hi = _mm_mulhi_epi16(vxi23x01234567, vxk23x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp23x01234567lo, vp23x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp23x01234567lo, vp23x01234567hi));

        const __m128i vi24x01234567 = _mm_loadl_epi64((const __m128i*) i24);
        const __m128i vk24x01234567 = _mm_loadl_epi64((const __m128i*) (k + 576));
        i24 += 8;

        const __m128i vxi24x01234567 = _mm_unpacklo_epi8(vi24x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi24x01234567));
        const __m128i vxk24x01234567 = _mm_unpacklo_epi8(vk24x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vk24x01234567));

        const __m128i vp24x01234567lo = _mm_mullo_epi16(vxi24x01234567, vxk24x01234567);
        const __m128i vp24x01234567hi = _mm_mulhi_epi16(vxi24x01234567, vxk24x01234567);

        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vp24x01234567lo, vp24x01234567hi));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpackhi_epi16(vp24x01234567lo, vp24x01234567hi));

        k += 8;

        const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.multiplier);
        const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.rounding);

        const __m128i vnmask0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123);
        const __m128i vnmask4567 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567);

        const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
        const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

        const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
        const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
        const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
        const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

        const __m128i vnmask02 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128i vnmask13 = _mm_shuffle_epi32(vnmask0123, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128i vnmask46 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128i vnmask57 = _mm_shuffle_epi32(vnmask4567, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod02 = _mm_sub_epi64(_mm_xor_si128(vabsprod02, vnmask02), vnmask02);
        const __m128i vprod13 = _mm_sub_epi64(_mm_xor_si128(vabsprod13, vnmask13), vnmask13);
        const __m128i vprod46 = _mm_sub_epi64(_mm_xor_si128(vabsprod46, vnmask46), vnmask46);
        const __m128i vprod57 = _mm_sub_epi64(_mm_xor_si128(vabsprod57, vnmask57), vnmask57);

        const __m128i vq31prod02 = _mm_srli_epi64(_mm_add_epi64(vprod02, vrounding), 31);
        const __m128i vq31prod13 = _mm_srli_epi64(_mm_add_epi64(vprod13, vrounding), 31);
        const __m128i vq31prod46 = _mm_srli_epi64(_mm_add_epi64(vprod46, vrounding), 31);
        const __m128i vq31prod57 = _mm_srli_epi64(_mm_add_epi64(vprod57, vrounding), 31);

        const __m128i vq31prod0213 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod02), _mm_castsi128_ps(vq31prod13), _MM_SHUFFLE(2, 0, 2, 0)));
        const __m128i vq31prod4657 = _mm_castps_si128(_mm_shuffle_ps(
            _mm_castsi128_ps(vq31prod46), _mm_castsi128_ps(vq31prod57), _MM_SHUFFLE(2, 0, 2, 0)));

        const __m128i vq31prod0123 = _mm_shuffle_epi32(vq31prod0213, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vq31prod4567 = _mm_shuffle_epi32(vq31prod4657, _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_mask);
        const __m128i vrem0123 =
          _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));
        const __m128i vrem4567 =
          _mm_add_epi32(_mm_and_si128(vq31prod4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod4567));

        const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.remainder_threshold);
        const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse2.shift);
        vacc0123 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
        vacc4567 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));

        w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t));

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_zero_point);
        __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

        vout01234567 = _mm_max_epi16(vout01234567, _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_min));
        vout01234567 = _mm_min_epi16(vout01234567, _mm_load_si128((const __m128i*) params->gemmlowp_sse2.output_max));

        __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) output, vout0123456701234567);
          output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
            vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
            output += 4;
          }
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
            vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int32_t) _mm_cvtsi128_si32(vout0123456701234567);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
