//
//  quant_weights_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/quant_weights.hpp>: block_weights is a trivially copyable view whose rows are the right
//  blocks, for blocks of 32 and of 256, and the quantised kernels' host side accepts it for each format core has. No
//  device needed.
//

#include <snicholls/gpu/quant_weights.hpp>

#include <snicholls/gpu/detail/quant_source.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace gpu = snicholls::gpu;

struct block32 {
    static constexpr std::size_t elements = 32;
    std::array<std::uint8_t, 18> bytes;
};
struct block256 {
    static constexpr std::size_t elements = 256;
    std::array<std::uint8_t, 144> bytes;
};

static_assert(std::is_trivially_copyable_v<gpu::block_weights<block32>> && std::is_aggregate_v<gpu::block_weights<block32>>);

TEST(GpuQuantWeights, RowsAreConsecutiveRunsOfBlocks)
{
    std::vector<block32> b(3 * 4);
    const gpu::block_weights<block32> w{b, 3, 128};
    EXPECT_EQ(w.row(0).data(), b.data());
    EXPECT_EQ(w.row(2).data(), b.data() + 8);
    EXPECT_EQ(w.row(1).size(), 4u);
    std::vector<block256> k(2 * 3);
    const gpu::block_weights<block256> wk{k, 2, 768};
    EXPECT_EQ(wk.row(1).data(), k.data() + 3);
    EXPECT_EQ(wk.row(1).size(), 3u);
}

TEST(GpuQuantWeights, TheKernelsCheckEachFormatsShape)
{
    namespace q = snicholls::gpu::detail::quant;
    for (std::size_t f = 0; f < q::format_count; ++f) {
        const std::size_t e = q::block_elements[f], bytes = q::block_bytes[f];
        if (q::format(f) == q::format::nf4)
            continue;   // its scales are apart; covered by the device tests
        const std::size_t rows = 3, cols = 2 * e;
        std::vector<std::byte> storage(rows * cols / e * bytes);
        q::parts p{q::format(f), storage, {}, {}, 0.0f, rows, cols};
        EXPECT_TRUE(q::valid(p)) << q::names[f];
        p.weights = std::span<const std::byte>(storage).first(storage.size() - 1);
        EXPECT_FALSE(q::valid(p)) << q::names[f];
        if (e > 1) {
            p.weights = storage;
            p.cols = cols + 1;
            EXPECT_FALSE(q::valid(p)) << q::names[f];
        }
    }
}

#ifdef SNICHOLLS_GPU_QUANT_K
TEST(GpuQuantWeights, CoresBlocksHaveTheSizesTheDeviceReads)
{
    namespace q = snicholls::gpu::detail::quant;
    EXPECT_EQ(q::traits<gpu::block_weights<snicholls::ml::block_q4_k>>::fmt, q::format::q4_k);
    EXPECT_EQ(q::traits<gpu::block_weights<snicholls::ml::block_tq1_0>>::fmt, q::format::tq1_0);
    static_assert(sizeof(snicholls::ml::block_q6_k) == 210 && sizeof(snicholls::ml::block_q2_k) == 84);
}
#endif

} // namespace
