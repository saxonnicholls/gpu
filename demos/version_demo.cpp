//
//  version_demo.cpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/gpu/version.hpp>: prints both versions and checks them.
//

#include <snicholls/gpu/version.hpp>

#include <cstdio>
#include <string>

int main()
{
    //! [version]
    SNICHOLLS_GPU_REQUIRE_VERSION(0, 1, 0);   // fails to compile against an older snicholls-gpu
    const std::string gpu(snicholls::gpu::version_string());
    const std::string core(snicholls::gpu::core_version_string());
    std::printf("snicholls-gpu %s, built against snicholls-core %s\n", gpu.c_str(), core.c_str());
    //! [version]
    if (gpu != SNICHOLLS_GPU_VERSION_STRING || snicholls::core::version() < SNICHOLLS_CORE_VERSION_CHECK(0, 12, 0)) {
        std::printf("version_demo: FAILED - expected core 0.12.0 or newer\n");
        return 1;
    }
    std::printf("version_demo: ok\n");
    return 0;
}
