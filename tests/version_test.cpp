//
//  version_test.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/gpu/version.hpp>.
//

#include <snicholls/gpu/version.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

namespace gv = snicholls::gpu;

static_assert(gv::version() == gv::version_major() * 10000 + gv::version_minor() * 100 + gv::version_patch());
static_assert(noexcept(gv::version_string()));
SNICHOLLS_GPU_REQUIRE_VERSION(0, 1, 0);
SNICHOLLS_CORE_REQUIRE_VERSION(0, 13, 0);   // the minimum, as snicholls_core_require(0.13.0) in CMakeLists.txt

TEST(GpuVersion, StringIsMajorDotMinorDotPatch)
{
    const std::string expected = std::to_string(gv::version_major()) + "." + std::to_string(gv::version_minor()) + "." +
                                 std::to_string(gv::version_patch());
    EXPECT_EQ(gv::version_string(), expected);
}

TEST(GpuVersion, CoreVersionIsTheOneInTheBuild)
{
    EXPECT_EQ(gv::core_version_string(), snicholls::core::version_string());
    EXPECT_GE(snicholls::core::version(), SNICHOLLS_CORE_VERSION_CHECK(0, 13, 0));
}

} // namespace
