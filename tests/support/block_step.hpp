//
//  block_step.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Test and benchmark support (not part of the library): every parameter, activation and gradient of one transformer
//  block's training step, as host vectors mirrored in device memory, with core's views over either.
//

#ifndef snicholls_gpu_tests_support_block_step_hpp
#define snicholls_gpu_tests_support_block_step_hpp

#include <snicholls/gpu/cuda.hpp>

#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace snicholls::gpu::testing {

namespace ml = snicholls::ml;
namespace gpu = snicholls::gpu;

inline std::vector<float> stream(std::size_t n, std::uint64_t seed, float scale = 1.0f, float offset = 0.0f)
{
    std::vector<float> v(n);
    snicholls::stats::wyrand g(seed);
    for (float& e : v)
        e = offset + scale * static_cast<float>(2 * snicholls::stats::uniform_real(g) - 1);
    return v;
}

// A host vector mirrored in device memory
struct mirror {
    std::vector<float> host;
    std::unique_ptr<gpu::device_buffer<float>> dev;
    const gpu::cuda_context* context;
    mirror(const gpu::cuda_context& c, std::vector<float> h)
        : host(std::move(h)), dev(std::make_unique<gpu::device_buffer<float>>(c, host.size())), context(&c)
    {
        dev->upload(host);
    }
    std::span<float> d() const { return dev->span(); }
    std::vector<float> back() const
    {
        std::vector<float> out(host.size());
        dev->download(out);
        return out;
    }
};


// Every parameter and buffer of one block's training step over S sequences of N tokens end to end, mirrored on the
// device; n is every row of the batch
template <ml::transformer_config C, std::size_t N = 24, std::size_t S = 1>
struct step {
    static constexpr std::size_t tokens = N, sequences = S;
    static constexpr std::size_t n = N * S, d = C.d_model, f = C.d_ff, qn = C.q_size(), kn = C.kv_size();
    static constexpr bool layer = C.norm == ml::norm_kind::layer, swiglu = C.mlp == ml::mlp_kind::swiglu, qk = C.qk_norm != ml::qk_norm_kind::none,
                          bias = C.bias;
    std::vector<mirror> m;
    enum { n1w, n1b, wq, wk, wv, wo, bq, bk, bv, bo, nq, nk, n2w, n2b, up, gate, down, bup, bgate, bdown, xs, dxs, count_params };
    enum { x0 = count_params, h1, x1, h2, q, k, v, qr, kr, y, lse, aup, agate, hidden, gn1w, gn1b, gn2w, gn2b, gq, gk, gv, gwq, gwk, gwv, gwo,
           gbq, gbk, gbv, gbo, gnq, gnk, gup, ggate, gdown, gbup, gbgate, gbdown, scratch, dy, mscratch, count };
    explicit step(const gpu::cuda_context& context)
    {
        auto p = [&](std::size_t size, std::uint64_t seed, float scale, float offset = 0.0f) { m.emplace_back(context, stream(size, seed, scale, offset)); };
        auto z = [&](std::size_t size) { m.emplace_back(context, std::vector<float>(size)); };
        p(d, 1, 0.2f, 1.0f), layer ? p(d, 2, 0.1f) : z(0), p(qn * d, 3, 0.3f), p(kn * d, 4, 0.3f), p(kn * d, 5, 0.3f), p(d * qn, 6, 0.2f);
        bias ? p(qn, 7, 0.1f) : z(0), bias ? p(kn, 8, 0.1f) : z(0), bias ? p(kn, 9, 0.1f) : z(0), bias ? p(d, 10, 0.1f) : z(0);
        qk ? p(C.head_dim, 11, 0.1f, 1.0f) : z(0), qk ? p(C.head_dim, 12, 0.1f, 1.0f) : z(0);
        p(d, 13, 0.2f, 1.0f), layer ? p(d, 14, 0.1f) : z(0), p(f * d, 15, 0.3f), swiglu ? p(f * d, 16, 0.3f) : z(0), p(d * f, 17, 0.1f);
        bias ? p(f, 18, 0.1f) : z(0), (bias && swiglu) ? p(f, 19, 0.1f) : z(0), bias ? p(d, 20, 0.1f) : z(0), p(n * d, 21, 1.5f), p(n * d, 22, 1.0f);
        z(n * d), z(n * d), z(n * d), z(n * d), z(n * qn), z(n * kn), z(n * kn), z(qk ? n * qn : 0), z(qk ? n * kn : 0), z(n * qn), z(n * C.n_heads);
        z(n * f), z(swiglu ? n * f : 0), z(n * f);
        // The gradients, in the order of the second enumeration: the norms', Q, K, V's, the projections', the biases',
        // the QK norms', the MLP's; then the working spaces
        auto size = [&](int i) { return m[std::size_t(i)].host.size(); };
        z(size(n1w)), z(size(n1b)), z(size(n2w)), z(size(n2b)), z(n * qn), z(n * kn), z(n * kn);
        z(size(wq)), z(size(wk)), z(size(wv)), z(size(wo)), z(size(bq)), z(size(bk)), z(size(bv)), z(size(bo)), z(size(nq)), z(size(nk));
        z(size(up)), z(size(gate)), z(size(down)), z(size(bup)), z(size(bgate)), z(size(bdown));
        z(n * d), z(n * qn), z(2 * n * f);
        if (m.size() != std::size_t(count))
            std::abort();   // the enumeration and the buffers must agree
    }
    // Views over device memory (dev = true) or the host copies
    std::span<float> s(int i, bool dev) { return dev ? m[std::size_t(i)].d() : std::span<float>(m[std::size_t(i)].host); }
    std::span<const float> c(int i, bool dev) { return s(i, dev); }
    ml::block_weights<ml::dense_weights<float>> weights(bool dev)
    {
        using W = ml::dense_weights<float>;
        ml::block_weights<W> w{};
        w.norm1_weight = c(n1w, dev), w.norm1_bias = c(n1b, dev), w.norm2_weight = c(n2w, dev), w.norm2_bias = c(n2b, dev);
        w.attention = {W{c(wq, dev), qn, d}, W{c(wk, dev), kn, d}, W{c(wv, dev), kn, d}, W{c(wo, dev), d, qn}, c(bq, dev), c(bk, dev), c(bv, dev),
                       c(bo, dev), c(nq, dev), c(nk, dev)};
        w.mlp = {W{c(up, dev), f, d}, swiglu ? W{c(gate, dev), f, d} : W{}, W{c(down, dev), d, f}, c(bup, dev), c(bgate, dev), c(bdown, dev)};
        return w;
    }
    ml::block_activations<float> act(bool dev)
    {
        return {s(x0, dev), s(h1, dev), s(x1, dev), s(h2, dev), {s(q, dev), s(k, dev), s(v, dev), s(qr, dev), s(kr, dev), s(y, dev), s(lse, dev)},
                s(aup, dev), s(agate, dev), s(hidden, dev)};
    }
    // Sequence i alone: every per-row buffer's slice, the weights' gradients shared
    ml::block_activations<float> act(bool dev, std::size_t i)
    {
        auto a = act(dev);
        for (std::span<float>* f : {&a.x0, &a.h1, &a.x1, &a.h2, &a.attention.q, &a.attention.k, &a.attention.v, &a.attention.q_raw,
                                    &a.attention.k_raw, &a.attention.y, &a.attention.lse, &a.mlp_up, &a.mlp_gate, &a.mlp_hidden})
            *f = f->empty() ? *f : f->subspan(i * f->size() / S, f->size() / S);
        return a;
    }
    ml::block_gradients<float> grads(bool dev, std::size_t i)
    {
        auto g = grads(dev);
        for (std::span<float>* f : {&g.attention.q, &g.attention.k, &g.attention.v, &g.attention.y, &g.mlp.scratch, &g.scratch})
            *f = f->subspan(i * f->size() / S, f->size() / S);
        return g;
    }
    ml::block_gradients<float> grads(bool dev)
    {
        return {s(gn1w, dev), s(gn1b, dev), s(gn2w, dev), s(gn2b, dev),
                {s(gq, dev), s(gk, dev), s(gv, dev), s(gwq, dev), s(gwk, dev), s(gwv, dev), s(gwo, dev), s(gbq, dev), s(gbk, dev), s(gbv, dev),
                 s(gbo, dev), s(gnq, dev), s(gnk, dev), s(dy, dev)},
                {s(gup, dev), s(ggate, dev), s(gdown, dev), s(gbup, dev), s(gbgate, dev), s(gbdown, dev), s(mscratch, dev)},
                s(scratch, dev)};
    }
};


} // namespace snicholls::gpu::testing

#endif /* snicholls_gpu_tests_support_block_step_hpp */
