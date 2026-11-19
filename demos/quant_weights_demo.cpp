//
//  quant_weights_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/quant_weights.hpp>: a view of a matrix in blocks of 256 (the shape of ggml's K-quants), and
//  its rows. Exits non-zero if a row is not the blocks it documents.
//

#include <snicholls/gpu/quant_weights.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
// A stand-in for a 256-number block of 144 bytes, as Q4_K is
struct block {
    static constexpr std::size_t elements = 256;
    std::array<std::uint8_t, 144> bytes;
};
} // namespace

int main()
{
    std::printf("snicholls-gpu quant_weights demo\n");
    //! [view]
    // 4 rows of 1024 inputs: 4 blocks a row, 16 in all, row by row
    std::vector<block> blocks(16);
    const snicholls::gpu::block_weights<block> w{blocks, 4, 1024};
    const auto third = w.row(2);   // blocks 8 to 11
    //! [view]
    const bool ok = third.size() == 4 && third.data() == blocks.data() + 8;
    std::printf("  row 2: %zu blocks from block %td\n", third.size(), third.data() - blocks.data());
    std::printf(ok ? "quant_weights demo: ok\n" : "quant_weights demo: FAILED\n");
    return ok ? 0 : 1;
}
