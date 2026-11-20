//
//  quant_weights.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::block_weights<Block>`: a weight matrix of any of ggml's quantised blocks, row by row, as the quantised-weight kernels read it.
/// \ingroup gpu
///
/// **Overview.** Core's `ml::quantized_weights<Block>` holds the blocks of 32 that pair with Q8_0 activations (Q8_0,
/// Q4_0, MXFP4). The other ggml formats the device kernels read (Q4_1, Q5_0, Q5_1, NVFP4, Q1_0, Q2_0, the K-quants Q2_K to Q6_K of 256,
/// the ternary TQ1_0 and TQ2_0, the IQ family) need the same view without that pairing: `rows` rows of `cols / Block::elements`
/// blocks, end to end, over the caller's memory (host memory for `opencl_quant.hpp`, device memory for
/// `cuda_quant.hpp`).
///
/// \snippet quant_weights_demo.cpp view
///
/// **Relationships.** `opencl_quant.hpp` and `cuda_quant.hpp` (the products over it), core's `ml/quantize.hpp`,
/// `ml/quant_k.hpp`, `ml/quant_ternary.hpp` (the blocks, their quantisers and dequantisers). Group: \ref gpu. It stands
/// in for a generic view in core, where it belongs once core has one.
///
/// **Guarantees.** Trivially copyable, two words and the shape; owns nothing.
///
/// **Errors.** None here; the products check the shape (`cols` a multiple of the block, the span's size) and throw
/// `std::invalid_argument`.
///
/// **Performance.** Not benchmarked: a view.
///
/// **Tests and demos.** `tests/quant_weights_test.cpp`, `demos/quant_weights_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<span>`.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_gpu_quant_weights_hpp
#define snicholls_gpu_quant_weights_hpp

#include <cstddef>
#include <span>

namespace snicholls::gpu {

/// \brief A quantised weight matrix over the caller's memory: `rows` rows of `cols / Block::elements` blocks, end to end.
/// \tparam Block A ggml block with `elements`, the numbers in one block.
template <typename Block>
struct block_weights {
    std::span<const Block> blocks{};   ///< `rows · cols / Block::elements` blocks, row by row.
    std::size_t rows = 0;              ///< The outputs.
    std::size_t cols = 0;              ///< The inputs: a multiple of `Block::elements`.

    /// \brief Row `i`'s blocks. \param i The row, below `rows`. \return Its `cols / Block::elements` blocks.
    [[nodiscard]] constexpr std::span<const Block> row(std::size_t i) const
    {
        return blocks.subspan(i * (cols / Block::elements), cols / Block::elements);
    }
};

} // namespace snicholls::gpu

#endif /* snicholls_gpu_quant_weights_hpp */
