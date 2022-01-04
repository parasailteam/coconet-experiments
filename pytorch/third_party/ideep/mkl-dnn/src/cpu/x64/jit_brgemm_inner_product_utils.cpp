/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_brgemm_inner_product_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

namespace brgemm_inner_product_utils {

format_tag_t get_brgemm_ip_weights_tag(
        cpu_isa_t isa, dim_t oc, data_type_t wei_dt, int n_sp_dims) {
    using namespace format_tag;
    const bool is_amx_int8 = isa == avx512_core_bf16_amx_int8;
    if (oc >= 64) {
        switch (wei_dt) {
            case data_type::f32:
                return pick(n_sp_dims, OI16i64o, OIw16i64o, OIhw16i64o,
                        OIdhw16i64o);
            case data_type::bf16:
                return pick(n_sp_dims, OI8i64o2i, OIw8i64o2i, OIhw8i64o2i,
                        OIdhw8i64o2i);
            case data_type::s8:
                return (!is_amx_int8) ? pick(n_sp_dims, OI4i64o4i, OIw4i64o4i,
                               OIhw4i64o4i, OIdhw4i64o4i)
                                      : pick(n_sp_dims, OI16i64o4i, OIw16i64o4i,
                                              OIhw16i64o4i, OIdhw16i64o4i);
            default: return format_tag::undef;
        }
    } else if (oc >= 32) {
        switch (wei_dt) {
            case data_type::f32:
                return pick(n_sp_dims, OI16i32o, OIw16i32o, OIhw16i32o,
                        OIdhw16i32o);
            case data_type::bf16:
                return pick(n_sp_dims, OI8i32o2i, OIw8i32o2i, OIhw8i32o2i,
                        OIdhw8i32o2i);
            case data_type::s8:
                return (!is_amx_int8) ? pick(n_sp_dims, OI4i32o4i, OIw4i32o4i,
                               OIhw4i32o4i, OIdhw4i32o4i)
                                      : pick(n_sp_dims, OI16i32o4i, OIw16i32o4i,
                                              OIhw16i32o4i, OIdhw16i32o4i);
            default: return format_tag::undef;
        }
    } else {
        switch (wei_dt) {
            case data_type::f32:
                return pick(n_sp_dims, OI16i16o, OIw16i16o, OIhw16i16o,
                        OIdhw16i16o);
            case data_type::bf16:
                return pick(n_sp_dims, OI8i16o2i, OIw8i16o2i, OIhw8i16o2i,
                        OIdhw8i16o2i);
            case data_type::s8:
                return (!is_amx_int8) ? pick(n_sp_dims, OI4i16o4i, OIw4i16o4i,
                               OIhw4i16o4i, OIdhw4i16o4i)
                                      : pick(n_sp_dims, OI16i16o4i, OIw16i16o4i,
                                              OIhw16i16o4i, OIdhw16i16o4i);
            default: return format_tag::undef;
        }
    }
}

// TODO: add support of post-ops with multiple binary and eltwise execution
bool post_ops_ok(
        jit_brgemm_primitive_conf_t &jbgp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len()) {
        case 0: return true;
        case 1: return is_eltwise(0) || p.contain(sum, 0);
        case 2:
            return (p.contain(sum, 0) && is_eltwise(1))
                    || (one_of(jbgp.src_dt, u8, s8) && p.contain(sum, 1)
                            && is_eltwise(0));
        default: return false;
    }

    return false;
}

status_t init_ip_conf_fwd(
        jit_brgemm_primitive_conf_t &jbgp, const primitive_attr_t &attr) {
    const bool is_amx_int8 = jbgp.isa == avx512_core_bf16_amx_int8;
    const bool is_int8 = one_of(jbgp.src_dt, u8, s8) && jbgp.wei_dt == s8;
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const auto &p = attr.post_ops_;
    jbgp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jbgp.with_eltwise = eltwise_ind != -1;
    if (jbgp.with_eltwise) jbgp.eltwise = p.entry_[eltwise_ind].eltwise;
    if (!post_ops_ok(jbgp, attr)) return status::unimplemented;
    if (jbgp.with_scales) {
        const auto &oscales = attr.output_scales_;
        jbgp.is_oc_scale = oscales.mask_ == 1 << 1;

        // only common and per-oc-channel scales are supported
        const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
        if (!oscales_ok) return status::unimplemented;
    }

    jbgp.use_buffer = IMPLICATION(jbgp.dst_dt == jbgp.acc_dt, jbgp.with_sum);

    constexpr int amx_int8_row = 64;
    jbgp.ic_block = (is_amx_int8) ? amx_int8_row : jbgp.simd_w;

    // gemm-based inner product performs better when oc = 1
    if (is_f32 && jbgp.oc == 1) return status::unimplemented;

    if (jbgp.oc >= 64) {
        jbgp.oc_block = 64;
    } else if (jbgp.oc >= 32) {
        jbgp.oc_block = 32;
    } else {
        jbgp.oc_block = 16;
    }

    jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);
    jbgp.os = jbgp.mb;

    // Configure matrix sizes
    static const int min_M = is_amx_int8 ? 16 : 6;
    int max_M = 64;
    // Work done by each thread is given by:
    //     (nb_oc / nb_oc_blocking) * (nb_os / nb_os_blocking)
    // As a first approximation we take nb_oc_blocking = nb_os_blocking = 1
    // Furthermore, we recall that
    //     nb_oc = oc / oc_block
    //     nb_os = os / os_block
    //
    // For f32 data type our objective is to determine the optimal value
    // of os_block such that the work amount per thread ~ 2
    if (is_f32) {
        const bool small_work_amt_per_thread
                = div_up(jbgp.os, max_M) * jbgp.nb_oc < 1.8f * jbgp.nthr;
        if (small_work_amt_per_thread)
            max_M = saturate(
                    16, max_M, div_up(jbgp.os * jbgp.nb_oc, 2 * jbgp.nthr));
    }

    jbgp.os_block = 1;
    for (int m_ = max_M; m_ >= min_M; m_--) {
        if (jbgp.os % m_ == 0) {
            jbgp.os_block = m_;
            break;
        }
    }
    if (jbgp.os_block == 1) jbgp.os_block = nstl::min(jbgp.os, max_M);

    jbgp.nb_oc_blocking = 1;
    const int small_oc_threshold = 256;
    const int small_os_threshold = 8;
    if (jbgp.os <= small_os_threshold && jbgp.oc <= small_oc_threshold) {
        // For small problems compute all oc blocks one chunck to avoid
        // parallel section
        jbgp.nb_oc_blocking = jbgp.nb_oc;
    }

    jbgp.nb_ic_blocking = 1;
    const int max_nb_ic_blocking = nstl::min(64, jbgp.nb_ic);
    if (IMPLICATION(!is_int8, jbgp.ic <= max_nb_ic_blocking * jbgp.ic_block)
            && everyone_is(1, jbgp.kw, jbgp.kh, jbgp.kd)) {
        // Optimization: data & weights layouts allow to generate
        // brgemm kernel with K = ic & batch = 1
        // (K = rnd_dn(ic, ic_block), K_tail = ic % ic_block & batch = 1)
        // instead of K = ic_block & batch = nb_ic_blocking
        jbgp.K = jbgp.ic <= jbgp.ic_block ? jbgp.ic
                                          : rnd_dn(jbgp.ic, jbgp.ic_block);
        jbgp.nb_ic_blocking = jbgp.nb_ic;
        jbgp.gemm_batch_size = 1;
    } else {
        jbgp.nb_ic_blocking = max_div(jbgp.nb_ic, max_nb_ic_blocking);
        if (jbgp.nb_ic_blocking == 1) jbgp.nb_ic_blocking = max_nb_ic_blocking;
        jbgp.gemm_batch_size = jbgp.nb_ic_blocking;

        jbgp.K = jbgp.ic_block;
    }

    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jbgp.adjusted_batch_size
            = div_up(rnd_up(jbgp.gemm_batch_size * sc_size, 4096), sc_size);

    jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);
    jbgp.nb_os_blocking = 1;
    // Work done by each thread is given by:
    //     (nb_oc / nb_oc_blocking) * (nb_os / nb_os_blocking)
    // For f32 data type we want to increase the nb_os_blocking such that
    //   * 1 <= nb_os_blocking <= 8 AND nb_os_blocking <= nb_os
    //   * Work amount per thread ~ 2
    //   * NOTE: here nb_oc_blocking = 1 as os is large
    if (jbgp.os > 256 && is_f32)
        jbgp.nb_os_blocking = saturate(1, nstl::min(8, jbgp.nb_os),
                nstl::min(nstl::max(jbgp.oc / jbgp.os / 2, 1),
                        div_up(jbgp.nb_os * jbgp.nb_oc, 2 * jbgp.nthr)));

    jbgp.M = jbgp.os_block;
    jbgp.M_tail = jbgp.os % jbgp.os_block;

    jbgp.N = jbgp.oc_block;
    jbgp.N_tail = jbgp.oc % jbgp.oc_block;
    jbgp.K_tail = jbgp.ic % jbgp.ic_block;

    jbgp.LDA = jbgp.ic_without_padding;
    jbgp.LDB = jbgp.N;
    jbgp.LDC = (jbgp.use_buffer) ? jbgp.N : jbgp.oc_without_padding;
    jbgp.LDD = jbgp.oc_without_padding;

    return status::success;
}

status_t init_ip_conf_bwd_d(jit_brgemm_primitive_conf_t &jbgp) {
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);

    jbgp.use_buffer_b = true;
    jbgp.use_buffer = jbgp.src_dt != jbgp.acc_dt;
    jbgp.ip_bwd_d_global_b_transpose = false;

    jbgp.oc_block = jbgp.simd_w;
    jbgp.ic_block
            = (jbgp.ic >= (is_f32 ? 512 : 64)) ? 64 : jbgp.ic >= 32 ? 32 : 16;

    jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_ic_blocking = 1;
    jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);
    jbgp.os = jbgp.mb;

    // Configure matrix sizes
    const int max_M = nstl::min(64, jbgp.os), min_M = 6;
    jbgp.os_block = 1;
    for (int m_ = max_M; m_ >= min_M; m_--) {
        if (jbgp.os % m_ == 0) {
            jbgp.os_block = m_;
            break;
        }
    }
    if (jbgp.os_block == 1) jbgp.os_block = nstl::min(jbgp.os, max_M);
    jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);
    jbgp.nb_os_blocking = 1;
    int os_blocking_max = 2;
    for (int bl = os_blocking_max; bl >= 1; bl--)
        if (jbgp.nb_os % bl == 0) {
            jbgp.nb_os_blocking = bl;
            break;
        }

    jbgp.M = jbgp.os_block;
    jbgp.M_tail = jbgp.os % jbgp.os_block;

    jbgp.K = jbgp.oc_block;
    jbgp.N = jbgp.ic_block;
    jbgp.N_tail = jbgp.ic % jbgp.ic_block;
    jbgp.K_tail = jbgp.oc % jbgp.oc_block;

    jbgp.LDA = jbgp.oc_without_padding;
    jbgp.LDB = jbgp.N;
    jbgp.LDC = (jbgp.use_buffer) ? jbgp.N : jbgp.ic_without_padding;
    jbgp.LDD = jbgp.ic_without_padding;

    jbgp.nb_oc_blocking = 1;
    const int oc_chunk_max_size = 64;
    for (int bl = oc_chunk_max_size; bl >= 1; bl--)
        if (jbgp.nb_oc % bl == 0) {
            jbgp.nb_oc_blocking = bl;
            break;
        }

    jbgp.nthr_oc_b = 1;
    const int num_work_to_parallel = div_up(jbgp.nb_ic, jbgp.nb_ic_blocking)
            * div_up(jbgp.nb_os, jbgp.nb_os_blocking);
    // For f32 data type, use oc reduction if we have
    //   * very large output channels
    //   * small work amount available to each thread
    if (is_f32 && (num_work_to_parallel < 2 * jbgp.nthr || jbgp.oc > 1024)) {
        const int min_chunck_sz = 16;
        const int num_min_chunk_sz = div_up(jbgp.nb_oc, min_chunck_sz);
        float ratio = 0.5f * num_min_chunk_sz * jbgp.nb_os
                + (float)num_min_chunk_sz / jbgp.nb_ic + 0.5f;
        jbgp.nthr_oc_b
                = saturate(1, nstl::min(4, num_min_chunk_sz), int(ratio));
        if (jbgp.nthr_oc_b > 1) {
            jbgp.nb_oc_blocking = div_up(jbgp.nb_oc, jbgp.nthr_oc_b);
            jbgp.nb_oc_blocking
                    /= div_up(jbgp.nb_oc_blocking, oc_chunk_max_size);
        }
    }
    jbgp.gemm_batch_size = jbgp.nb_oc_blocking;
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jbgp.adjusted_batch_size
            = div_up(rnd_up(jbgp.gemm_batch_size * sc_size, 4096), sc_size);

    jbgp.use_buffer = jbgp.src_dt != jbgp.acc_dt || jbgp.nthr_oc_b > 1;

    return status::success;
}

void thread_balance(const jit_brgemm_primitive_conf_t &j, int &nb_os_blocking_,
        int &nthr_, int &nthr_mb_, int &nthr_oc_b_, int &nthr_ic_b_) {
    nthr_ = nthr_mb_ = nthr_oc_b_ = nthr_ic_b_ = 1;
    nb_os_blocking_ = j.nb_os_blocking;

    const bool is_f32 = everyone_is(f32, j.src_dt, j.wei_dt, j.dst_dt);
    const int max_threads = j.nthr;
    const int nthr = max_threads;
    int ic_chunks = j.nb_ic / j.nb_ic_blocking;
    int oc_chunks = j.nb_oc / j.nb_oc_blocking;
    auto calc_mem_cost = [=](int nb_os_blocking, int nthr_mb, int nthr_oc,
                                 int nthr_ic) {
        int src_size = j.ic * j.mb;
        int dst_size = j.oc * j.mb;
        int wei_size = j.ic * j.oc;
        int os_chunks = div_up(j.nb_os, nb_os_blocking);
        float wei_compensation_scale = 0.5f * (dst_size + src_size) / wei_size;
        float oi_channels_ratio = (float)src_size / dst_size;
        auto get_src_coef = [=]() {
            if (is_f32) {
                float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
                src_coef *= types::data_type_size(j.src_dt);
                src_coef *= 4 * saturate(1, 4, div_up(j.ic, 1024));
                if (wei_compensation_scale < 2.0f)
                    src_coef += sqrtf(2.0f / wei_compensation_scale);
                return src_coef;
            }
            float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
            src_coef *= 4 * types::data_type_size(j.src_dt);
            if (wei_compensation_scale < 1.0f) src_coef *= 4.0f;

            return src_coef;
        };

        auto get_dst_coef = [=]() {
            if (is_f32) {
                float dst_coef = types::data_type_size(j.dst_dt)
                        * nstl::max(oi_channels_ratio, 1.0f);
                return dst_coef;
            }

            return 2 * types::data_type_size(j.dst_dt)
                    * nstl::max(oi_channels_ratio, 1.0f);
        };

        auto get_wei_coef = [=]() {
            if (is_f32) {
                return nstl::max(
                        4.0f - j.mb / 2048 * wei_compensation_scale, 1.0f);
            }

            return nstl::max(wei_compensation_scale, 1.0f);
        };

        float src_tr = 0.0f;
        if (j.use_buffer_a && !is_f32) {
            int src_tr_oc_par_work = div_up(os_chunks, nthr_mb)
                    * div_up(ic_chunks, nthr_ic) * j.nb_ic_blocking;
            src_tr = get_src_coef() * div_up(src_tr_oc_par_work, nthr_oc)
                    * nb_os_blocking * j.os_block * j.ic_block;
        }

        float dst_tr = 0.0f;
        if (j.use_buffer_b && !is_f32) {
            int dst_tr_ic_par_work = div_up(os_chunks, nthr_mb)
                    * div_up(oc_chunks, nthr_oc) * j.nb_oc_blocking;
            dst_tr = get_dst_coef() * div_up(dst_tr_ic_par_work, nthr_ic)
                    * nb_os_blocking * j.os_block * j.oc_block;
        }

        float src_v = get_src_coef() * div_up(os_chunks, nthr_mb)
                * div_up(ic_chunks, nthr_ic) * nb_os_blocking * j.os_block
                * j.nb_ic_blocking * j.ic_block;
        float dst_v = get_dst_coef() * div_up(os_chunks, nthr_mb)
                * div_up(oc_chunks, nthr_oc) * nb_os_blocking * j.os_block
                * j.nb_oc_blocking * j.oc_block;

        auto acc_dt_sz = types::data_type_size(j.acc_dt);
        float wei_v = get_wei_coef() * acc_dt_sz * div_up(oc_chunks, nthr_oc)
                * div_up(ic_chunks, nthr_ic) * j.nb_oc_blocking * j.oc_block
                * j.nb_ic_blocking * j.ic_block;

        float wei_r = 0;
        if (nthr_mb > 1) {
            auto wei_dt_sz = types::data_type_size(j.wei_dt);
            int wei_r_mb_par_work = div_up(oc_chunks, nthr_oc)
                    * div_up(ic_chunks, nthr_ic) * j.nb_oc_blocking
                    * j.nb_ic_blocking;
            wei_r = get_wei_coef() * div_up(wei_r_mb_par_work, nthr_mb)
                    * j.oc_block * j.ic_block
                    * (wei_dt_sz
                            + (is_f32 ? div_up(j.os, 1024) : 1) * nthr_mb
                                    * acc_dt_sz);
        }

        return src_tr + dst_tr + src_v + dst_v + wei_v + wei_r;
    };

    float best_mem_cost
            = calc_mem_cost(nb_os_blocking_, nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* find the best thread distribution with lowest memory cost */
    const int min_osb_chunk = is_f32 ? 32 : 1;
    const int nthr_mb_max = nstl::min(nthr, div_up(j.nb_os, min_osb_chunk));
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        int nb_os_blocking = j.nb_os_blocking;
        int os_chunks = div_up(j.nb_os, nb_os_blocking);
        if (os_chunks < nthr_mb) {
            int coef = saturate(1, 4, 2 * j.mb / (j.oc + j.ic));
            int os_blocking_max = div_up(div_up(j.nb_os, coef), nthr_mb);
            for (int bl = os_blocking_max; bl >= 1; bl--)
                if (j.nb_os % bl == 0) {
                    nb_os_blocking = bl;
                    break;
                }
        }

        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, oc_chunks);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, ic_chunks);

            float mem_cost = calc_mem_cost(
                    nb_os_blocking, nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nb_os_blocking_ = nb_os_blocking;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    nthr_ = nthr_mb_ * nthr_oc_b_ * nthr_ic_b_;
}

status_t init_ip_conf_bwd_w(jit_brgemm_primitive_conf_t &jbgp) {
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);
    const bool big_ic_blk_ok
            = is_f32 && jbgp.ic % (4 * jbgp.simd_w) == 0 && jbgp.mb <= 128;
    jbgp.ic_block = big_ic_blk_ok ? 4 * jbgp.simd_w : jbgp.simd_w;
    if (jbgp.oc >= 64)
        jbgp.oc_block = 64;
    else if (jbgp.oc >= 32)
        jbgp.oc_block = 32;
    else
        jbgp.oc_block = 16;

    jbgp.nb_ic = div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_oc = div_up(jbgp.oc, jbgp.oc_block);
    jbgp.nb_oc_blocking = 1;
    jbgp.nb_ic_blocking = jbgp.nb_ic % 2 ? 1 : 2;

    jbgp.os = jbgp.mb;
    jbgp.os_block = 16;
    jbgp.nb_os = div_up(jbgp.os, jbgp.os_block);

    // Configure matrix sizes
    jbgp.M = jbgp.ic_block;
    jbgp.M_tail = jbgp.ic % jbgp.ic_block;

    jbgp.N = jbgp.oc_block;
    jbgp.N_tail = jbgp.oc % jbgp.oc_block;
    jbgp.K = jbgp.os_block;
    jbgp.K_tail = jbgp.os % jbgp.os_block;

    jbgp.nb_os_blocking = 1;
    int os_blocking_max = 64;
    for (int bl = os_blocking_max; bl >= 1; bl--)
        if (jbgp.nb_os % bl == 0) {
            jbgp.nb_os_blocking = bl;
            break;
        }

    int nb_os_blocking, nthr, nthr_mb, nthr_oc, nthr_ic;
    thread_balance(jbgp, nb_os_blocking, nthr, nthr_mb, nthr_oc, nthr_ic);

    jbgp.nb_os_blocking = nb_os_blocking;
    jbgp.nthr = nthr;
    jbgp.nthr_mb = nthr_mb;
    jbgp.nthr_oc_b = nthr_oc;
    jbgp.nthr_ic_b = nthr_ic;

    jbgp.gemm_batch_size = jbgp.nb_os_blocking;
    // to avoid cache concurrent write access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jbgp.adjusted_batch_size
            = div_up(rnd_up(jbgp.gemm_batch_size * sc_size, 4096), sc_size);

    jbgp.use_buffer = IMPLICATION(jbgp.wei_dt == jbgp.acc_dt, jbgp.nthr_mb > 1);
    jbgp.use_buffer_a = true;
    const bool is_oc_big_2_pow = jbgp.oc >= 512 && math::is_pow2(jbgp.oc);
    const bool is_huge_oc = jbgp.oc >= 4 * 1024;
    jbgp.use_buffer_b = jbgp.dst_dt == bf16 || is_oc_big_2_pow || is_huge_oc;

    jbgp.LDA = jbgp.K;
    jbgp.LDB = (jbgp.use_buffer_b) ? jbgp.N * jbgp.nb_oc_blocking
                                   : jbgp.oc_without_padding;
    jbgp.LDC = jbgp.LDD = jbgp.N;

    return status::success;
}

status_t init_ip_conf(cpu_isa_t isa, jit_brgemm_primitive_conf_t &jbgp,
        const inner_product_desc_t &ipd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    using namespace prop_kind;
    if (!mayiuse(avx512_common)) return status::unimplemented;

    int ndims = src_d.ndims();
    if (weights_d.ndims() != ndims || dst_d.ndims() != 2)
        return status::unimplemented;

    jbgp = zero<decltype(jbgp)>();
    jbgp.ndims = ndims;
    jbgp.isa = isa;
    jbgp.prop_kind = ipd.prop_kind;
    jbgp.ngroups = 1;
    jbgp.mb = src_d.dims()[0];
    jbgp.oc_without_padding = dst_d.dims()[1];
    jbgp.oc = jbgp.oc_without_padding;
    jbgp.ic_without_padding = src_d.dims()[1];
    jbgp.ic = jbgp.ic_without_padding;
    jbgp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jbgp.ih = (ndims < 4) ? 1 : src_d.dims()[ndims - 2];
    jbgp.iw = (ndims < 3) ? 1 : src_d.dims()[ndims - 1];
    jbgp.od = jbgp.oh = jbgp.ow = 1;
    jbgp.kd = (ndims == 5) ? weights_d.dims()[2] : 1;
    jbgp.kh = (ndims < 4) ? 1 : weights_d.dims()[ndims - 2];
    jbgp.kw = (ndims < 3) ? 1 : weights_d.dims()[ndims - 1];
    jbgp.stride_d = jbgp.stride_h = jbgp.stride_w = 1;

    if (!everyone_is(1, jbgp.ow, jbgp.oh, jbgp.od))
        return status::unimplemented;
    if (jbgp.kw != jbgp.iw || jbgp.kh != jbgp.ih || jbgp.kd != jbgp.id)
        return status::unimplemented;
    if (!everyone_is(1, jbgp.kw, jbgp.kh, jbgp.kd))
        return status::unimplemented;

    const int full_simd_w = 16;
    jbgp.simd_w = full_simd_w;

    jbgp.with_bias
            = pick_by_prop_kind(jbgp.prop_kind, ipd.bias_desc.format_kind,
                      format_kind::undef, ipd.diff_bias_desc.format_kind)
            != format_kind::undef;

    jbgp.src_dt = src_d.data_type();
    jbgp.dst_dt = dst_d.data_type();
    jbgp.wei_dt = weights_d.data_type();
    jbgp.bia_dt = jbgp.with_bias
            ? pick_by_prop_kind(jbgp.prop_kind, ipd.bias_desc.data_type,
                    data_type::undef, ipd.diff_bias_desc.data_type)
            : data_type::undef;
    jbgp.signed_input = isa == avx512_core_vnni && jbgp.src_dt == s8;
    const bool is_int8 = one_of(jbgp.src_dt, u8, s8) && jbgp.wei_dt == s8;
    const bool is_bf16
            = everyone_is(bf16, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt)
            || pick_by_prop_kind(jbgp.prop_kind,
                    everyone_is(bf16, jbgp.src_dt, jbgp.wei_dt)
                            && jbgp.dst_dt == f32,
                    everyone_is(bf16, jbgp.wei_dt, jbgp.dst_dt)
                            && jbgp.src_dt == f32,
                    everyone_is(bf16, jbgp.src_dt, jbgp.dst_dt)
                            && jbgp.wei_dt == f32);
    const bool is_f32 = everyone_is(f32, jbgp.src_dt, jbgp.wei_dt, jbgp.dst_dt);

    if (!IMPLICATION(is_int8,
                one_of(isa, avx512_core_vnni, avx512_core_bf16_amx_int8)))
        return status::unimplemented;
    if (!IMPLICATION(is_bf16, isa == avx512_core_bf16))
        return status::unimplemented;
    if (!IMPLICATION(is_f32, isa == avx512_core)) return status::unimplemented;

    if (is_int8) {
        jbgp.acc_dt = s32;
        jbgp.with_scales = true;
    } else if (is_bf16) {
        jbgp.acc_dt = f32;
    } else if (is_f32) {
        jbgp.acc_dt = f32;
    } else
        return status::unimplemented;

    auto set_or_check_tags = [&]() -> status_t {
        using namespace format_tag;
        format_tag_t desired_src_tag = pick(ndims - 2, nc, ncw, nchw, ncdhw);
        format_tag_t desired_dst_tag = nc;

        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, desired_src_tag));
            jbgp.src_tag = desired_src_tag;
        } else {
            jbgp.src_tag
                    = memory_desc_matches_one_of_tag(src_md, desired_src_tag);
        }

        if (dst_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(dst_md, desired_dst_tag));
            jbgp.dst_tag = desired_dst_tag;
        } else {
            jbgp.dst_tag = memory_desc_matches_one_of_tag(dst_md, nc);
        }

        if (one_of(format_tag::undef, jbgp.src_tag, jbgp.dst_tag))
            return status::unimplemented;

        if (jbgp.with_bias && bias_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));

        memory_desc_t want_wei_md = weights_md;
        jbgp.wei_tag = get_brgemm_ip_weights_tag(
                isa, (dim_t)jbgp.oc, jbgp.wei_dt, ndims - 2);
        CHECK(memory_desc_init_by_tag(want_wei_md, jbgp.wei_tag));

        if (jbgp.signed_input) {
            want_wei_md.extra.flags = 0
                    | memory_extra_flags::compensation_conv_s8s8
                    | memory_extra_flags::scale_adjust;
            want_wei_md.extra.compensation_mask = (1 << 0);
            want_wei_md.extra.scale_adjust
                    = platform::s8s8_weights_scale_factor();
        }
        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return status::success;
        }
        return (want_wei_md == weights_md) ? status::success
                                           : status::unimplemented;
    };

    CHECK(set_or_check_tags());

    jbgp.brg_type = brgemm_addr;
    jbgp.nthr = nthreads;

    switch (jbgp.prop_kind) {
        case forward_training:
        case forward_inference: return init_ip_conf_fwd(jbgp, attr);
        case backward_data: return init_ip_conf_bwd_d(jbgp);
        case backward_weights: return init_ip_conf_bwd_w(jbgp);
        default: assert(!"invalid prop_kind"); return invalid_arguments;
    }
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_primitive_conf_t &jbgp) {

    size_t sc_size = sizeof(brgemm_batch_element_t);
    size_t n_elems = jbgp.nthr * jbgp.adjusted_batch_size;
    if (jbgp.brg_type == brgemm_addr) {
        scratchpad.book(key_brgemm_primitive_batch, n_elems, sc_size, 64);
    }
    if (jbgp.use_buffer) {
        size_t nelements = (size_t)jbgp.nthr * jbgp.LDC * jbgp.M;
        if (jbgp.prop_kind == dnnl_backward_weights && jbgp.nthr_mb > 1) {
            int n_reduction_buffers = jbgp.nthr_mb - (jbgp.wei_dt == f32);
            nelements = (size_t)n_reduction_buffers * jbgp.nb_ic * jbgp.ic_block
                    * jbgp.nb_oc * jbgp.oc_block;
        } else if (jbgp.prop_kind == dnnl_backward_data && jbgp.nthr_oc_b > 1) {
            int n_reduction_buffers = jbgp.nthr_oc_b - (jbgp.dst_dt == f32);
            nelements = (size_t)n_reduction_buffers * jbgp.ic * jbgp.os;
        }
        scratchpad.book(key_brgemm_primitive_buffer, nelements,
                types::data_type_size(jbgp.acc_dt));
    }

    if (jbgp.use_buffer_a) {
        int ic_chunks = div_up(
                div_up(jbgp.nb_ic, jbgp.nb_ic_blocking), jbgp.nthr_ic_b);
        int os_chunks
                = div_up(div_up(jbgp.nb_os, jbgp.nb_os_blocking), jbgp.nthr_mb);
        scratchpad.book(key_brgemm_primitive_buffer_a,
                jbgp.nthr * ic_chunks * os_chunks * jbgp.gemm_batch_size
                        * jbgp.os_block * jbgp.ic_block * jbgp.nb_ic_blocking,
                types::data_type_size(jbgp.src_dt));
    }

    if (jbgp.use_buffer_b && jbgp.prop_kind == dnnl_backward_weights) {
        int os_chunks
                = div_up(div_up(jbgp.nb_os, jbgp.nb_os_blocking), jbgp.nthr_mb);
        scratchpad.book(key_brgemm_primitive_buffer_b,
                jbgp.nthr * os_chunks * jbgp.gemm_batch_size * jbgp.os_block
                        * jbgp.LDB,
                types::data_type_size(jbgp.dst_dt));
    }

    if (jbgp.use_buffer_b && jbgp.prop_kind == dnnl_backward_data) {
        int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
        if (!jbgp.ip_bwd_d_global_b_transpose)
            scratchpad.book(key_brgemm_primitive_buffer_b,
                    (dim_t)jbgp.nthr * jbgp.gemm_batch_size * size_B,
                    types::data_type_size(jbgp.wei_dt));
        else
            scratchpad.book(key_brgemm_primitive_buffer_b,
                    (dim_t)jbgp.nb_oc * jbgp.nb_ic * size_B,
                    types::data_type_size(jbgp.wei_dt));
    }

    if (jbgp.prop_kind == dnnl_backward_weights && jbgp.with_bias
            && (jbgp.bia_dt == bf16 || jbgp.nthr_mb > 1)) {
        int nbuffers = jbgp.nthr_mb - (jbgp.bia_dt == f32);
        scratchpad.book(key_iprod_bias_bf16_convert_wsp, nbuffers * jbgp.oc,
                types::data_type_size(jbgp.acc_dt));
    }

    if (dnnl_thr_syncable() && jbgp.prop_kind == dnnl_backward_weights)
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx, 1);

    if (jbgp.isa == avx512_core_bf16_amx_int8)
        scratchpad.book(
                key_conv_amx_tile_buffer, jbgp.nthr * 1024, sizeof(char));
}

} // namespace brgemm_inner_product_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
