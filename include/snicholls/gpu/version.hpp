//
//  version.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The snicholls-gpu version, and the core version it was built against.
/// \ingroup gpu
///
/// **Overview.** The version of this repository, written in three places that `scripts/check_version.py` keeps equal:
/// the macros below, `project(... VERSION)` in `CMakeLists.txt` and the release tag. `SNICHOLLS_GPU_REQUIRE_VERSION`
/// is the compile-time check for a consumer's header, like core's `SNICHOLLS_CORE_REQUIRE_VERSION`.
///
/// \snippet version_demo.cpp version
///
/// **Relationships.** `snicholls::core::version()` (core's own); the CMake check `snicholls_core_require`.
/// Group: \ref gpu.
///
/// **Guarantees.** `constexpr`, `noexcept`.
///
/// **Errors.** None; `SNICHOLLS_GPU_REQUIRE_VERSION` is a `static_assert`.
///
/// **Performance.** Compile time only.
///
/// **Tests and demos.** `tests/version_test.cpp`, `demos/version_demo.cpp`.
///
/// **Dependencies.** `<string_view>`, `<snicholls/core/version.hpp>`.
///
/// **Known issues.** None.
///
/// **Status.** Experimental, pre-1.0.

/// \defgroup gpu snicholls-gpu
/// \brief Backends of snicholls-core's interfaces on GPUs.

#ifndef snicholls_gpu_version_hpp
#define snicholls_gpu_version_hpp

#include <snicholls/core/version.hpp>

#include <string_view>

/// \brief The major version.
#define SNICHOLLS_GPU_VERSION_MAJOR 0
/// \brief The minor version.
#define SNICHOLLS_GPU_VERSION_MINOR 10
/// \brief The patch version.
#define SNICHOLLS_GPU_VERSION_PATCH 1

/// \brief The version as one comparable integer: major * 10000 + minor * 100 + patch.
#define SNICHOLLS_GPU_VERSION                                                                        \
    SNICHOLLS_CORE_VERSION_CHECK(SNICHOLLS_GPU_VERSION_MAJOR, SNICHOLLS_GPU_VERSION_MINOR, \
                                 SNICHOLLS_GPU_VERSION_PATCH)

/// \brief The version as a string literal, "major.minor.patch".
#define SNICHOLLS_GPU_VERSION_STRING                                  \
    SNICHOLLS_CORE_VERSION_STRINGIFY(SNICHOLLS_GPU_VERSION_MAJOR) "." \
    SNICHOLLS_CORE_VERSION_STRINGIFY(SNICHOLLS_GPU_VERSION_MINOR) "." \
    SNICHOLLS_CORE_VERSION_STRINGIFY(SNICHOLLS_GPU_VERSION_PATCH)

/// \brief A `static_assert` that snicholls-gpu is at least major.minor.patch.
#define SNICHOLLS_GPU_REQUIRE_VERSION(major, minor, patch)                                            \
    static_assert(SNICHOLLS_GPU_VERSION >= SNICHOLLS_CORE_VERSION_CHECK(major, minor, patch),         \
                  "snicholls-gpu " SNICHOLLS_GPU_VERSION_STRING " is older than the " #major "." #minor \
                  "." #patch " this code requires")

namespace snicholls::gpu {

/// \brief The major version. \return the number.
constexpr int version_major() noexcept { return SNICHOLLS_GPU_VERSION_MAJOR; }
/// \brief The minor version. \return the number.
constexpr int version_minor() noexcept { return SNICHOLLS_GPU_VERSION_MINOR; }
/// \brief The patch version. \return the number.
constexpr int version_patch() noexcept { return SNICHOLLS_GPU_VERSION_PATCH; }
/// \brief The version as one integer. \return major * 10000 + minor * 100 + patch.
constexpr int version() noexcept { return SNICHOLLS_GPU_VERSION; }
/// \brief The version as text. \return "major.minor.patch".
constexpr std::string_view version_string() noexcept { return SNICHOLLS_GPU_VERSION_STRING; }
/// \brief The snicholls-core version these headers were compiled against. \return core's version string.
constexpr std::string_view core_version_string() noexcept { return SNICHOLLS_CORE_VERSION_STRING; }

} // namespace snicholls::gpu

#endif /* snicholls_gpu_version_hpp */
