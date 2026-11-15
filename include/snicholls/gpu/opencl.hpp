//
//  opencl.hpp
//  snicholls-gpu
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::gpu::opencl_backend`: a `snicholls::linalg::blas_backend` whose matrix products run on an OpenCL device, a GPU by preference.
/// \ingroup gpu
///
/// **Overview.** snicholls-core states the linear algebra interface (`linalg::blas_backend`) and its reference
/// implementation on the CPU; this header is an implementation on a GPU, through OpenCL 1.2, so code written against
/// core's free functions (`linalg::gemm`, and through it `ml::linear_sequence` and the batched training passes) runs
/// on the device with no change but the backend it is handed.
///
/// - `opencl_context::create(choice)`: finds a device (a GPU first, or any OpenCL device), creates its context and
///   queue, and compiles the kernels; `result` with an `opencl_failure` when there is none or the build fails.
/// - `opencl_backend{&context}`: the backend, a pointer to the context, trivially copyable as the concept requires.
///   `gemm` and `gemv` run on the device; `dot`, `axpy`, `syrk` and `triangular_solve` run on the host through
///   core's `linalg::reference` (later versions move them).
///
/// Each product copies the spans its views cover to the device, runs one tiled kernel and copies the result's span
/// back, so it pays two transfers: it wins over the CPU only for large enough products (see Performance). Strided
/// views of every layout are read in place; the gaps of the result's span are preserved.
///
/// \snippet opencl_demo.cpp opencl
///
/// **Relationships.** `snicholls/core/linalg/backend.hpp` (the concept, from snicholls-core), core's
/// `linalg::reference` (what the device's results are checked against, bit for bit) and its conformance suite
/// (`snicholls/core/conformance/linalg.hpp`). Group: \ref gpu.
///
/// **Implementations.** One kernel for `gemm` (and `gemv` as a product with one column): 64 x 64 tiles of the result,
/// 16 deep, in local memory; 16 x 16 work-items each computing 4 x 4 outputs from registers; each tile loaded along
/// its operand's unit stride, so a transposed operand (`X Wᵀ`, `dYᵀ X`) is read as fast; each output summed in index
/// order with contraction off
/// (`#pragma OPENCL FP_CONTRACT OFF`), then `alpha s` or `alpha s + beta c`, the reference's arithmetic in the
/// reference's order. `double` runs on the device when it has `cl_khr_fp64`, else on the host by the reference.
///
/// **Guarantees.** On a device that keeps IEEE single precision (no flushing of subnormals), `gemm` and `gemv` give
/// the reference's bits. One context serves any number of threads: its kernel arguments and enqueues are under one
/// mutex per context. The context owns every OpenCL object and releases them when destroyed; a backend must not
/// outlive its context.
///
/// **Errors.** `opencl_context::create` returns an `opencl_failure` (no platform, no device, the program does not
/// build, or another OpenCL call failed, with its code and the build log). A product that fails on the device
/// (out of device memory, a lost device) throws `opencl_exception`, a `std::runtime_error` with the OpenCL code: an
/// unexpected failure, so not a `result`.
///
/// **Performance.** `benchmarks/opencl_benchmark.cpp`, the device against core's reference on the host, transfers
/// included; see `BENCHMARKS.md`.
///
/// **Tests and demos.** `tests/opencl_test.cpp` (core's linalg BLAS conformance suite, the reference's bits, strided
/// and transposed views, threads), `demos/opencl_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<memory>`, `<mutex>`, `<stdexcept>`, `<string>`, `<type_traits>`, `<vector>`, the
/// OpenCL 1.2 headers (`<OpenCL/opencl.h>` on macOS, `<CL/cl.h>` elsewhere), `snicholls/core/linalg/reference.hpp`,
/// `snicholls/core/result.hpp`. Link `OpenCL::OpenCL`.
///
/// **Known issues.** Apple has deprecated OpenCL since macOS 10.14 (it still runs, at 1.2); the header silences the
/// deprecation warnings. A device without subnormals (some GPUs flush them) agrees with the reference only to
/// rounding near the bottom of float's range.
///
/// **Status.** Experimental, new in 0.1.0.

#ifndef snicholls_gpu_opencl_hpp
#define snicholls_gpu_opencl_hpp

#ifndef CL_TARGET_OPENCL_VERSION
/// \brief The OpenCL headers' target: 1.2, what every supported device has (define it first to choose another).
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#ifndef CL_SILENCE_DEPRECATION
/// \brief Apple's OpenCL headers are deprecated since macOS 10.14; silence the warnings they would raise.
#define CL_SILENCE_DEPRECATION
#endif
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace snicholls::gpu {

/// \brief Which OpenCL device `opencl_context::create` takes.
enum class device_choice {
    gpu,   ///< A GPU, or fail.
    any,   ///< A GPU if there is one, else any device (a CPU implementation such as PoCL).
};

/// \brief Why `opencl_context::create` failed.
enum class opencl_error {
    no_platform,    ///< No OpenCL platform is installed.
    no_device,      ///< No device of the kind asked for.
    build_failed,   ///< The kernels did not compile; `opencl_failure::log` holds the build log.
    call_failed,    ///< Another OpenCL call failed; `opencl_failure::code` holds its code.
};

/// \brief A failure of `opencl_context::create`: what failed, the OpenCL code, and the build log if any.
struct opencl_failure {
    opencl_error kind = opencl_error::call_failed;   ///< What failed.
    cl_int code = CL_SUCCESS;                        ///< The OpenCL code.
    std::string log{};                               ///< The program's build log, for `build_failed`.
};

/// \brief A product that failed on the device: an unexpected failure (out of device memory, a lost device).
class opencl_exception : public std::runtime_error {
public:
    /// \brief An exception for a failed call.
    /// \param what The call.
    /// \param code Its OpenCL code.
    opencl_exception(const std::string& what, cl_int code)
        : std::runtime_error("snicholls::gpu: " + what + " failed with OpenCL error " + std::to_string(code)), code_(code)
    {
    }
    /// \brief The OpenCL code. \return The code.
    [[nodiscard]] cl_int code() const noexcept { return code_; }

private:
    cl_int code_;
};

namespace detail {

    // The kernels: C = alpha A B + beta C over strided views, 16 x 16 tiles, each output summed in index order with
    // contraction off, so the result is linalg::reference's
    inline constexpr const char* gemm_source = R"CLC(
#pragma OPENCL FP_CONTRACT OFF
// 64 x 64 tiles of C, 16 deep in k; a work-group of 16 x 16 items, each computing 4 x 4 outputs, every output summed
// in index order. A tile of an operand is loaded with consecutive items along its unit stride (row-major or
// column-major), so the loads coalesce whichever way the operand is laid out.
#define TSM 64
#define TSN 64
#define TSK 16
#define WPT 4
#define RTS 16
#define GEMM(NAME, T)                                                                                               \
__kernel __attribute__((reqd_work_group_size(RTS, RTS, 1)))                                                        \
void NAME(const long m, const long n, const long k, const T alpha,                                                 \
          __global const T* a, const long sa0, const long sa1, __global const T* b, const long sb0, const long sb1,  \
          const T beta, __global T* c, const long sc0, const long sc1)                                             \
{                                                                                                                  \
    __local T as[TSK][TSM];                                                                                        \
    __local T bs[TSK][TSN];                                                                                        \
    const int tc = get_local_id(0), tr = get_local_id(1), tid = tr * RTS + tc;                                     \
    const long col0 = (long)get_group_id(0) * TSN, row0 = (long)get_group_id(1) * TSM;                             \
    const bool a_by_rows = sa1 == 1 || sa0 != 1, b_by_rows = sb1 == 1 || sb0 != 1;                                 \
    T acc[WPT][WPT];                                                                                               \
    for (int i = 0; i < WPT; ++i)                                                                                  \
        for (int j = 0; j < WPT; ++j)                                                                              \
            acc[i][j] = (T)0;                                                                                      \
    for (long t = 0; t < k; t += TSK) {                                                                            \
        for (int l = 0; l < (TSM * TSK) / (RTS * RTS); ++l) {                                                      \
            const int e = tid + l * RTS * RTS;                                                                     \
            const int ar = a_by_rows ? e / TSK : e % TSM, ap = a_by_rows ? e % TSK : e / TSM;                       \
            const long gr = row0 + ar, gp = t + ap;                                                                \
            as[ap][ar] = (gr < m && gp < k) ? a[gr * sa0 + gp * sa1] : (T)0;                                       \
            const int bp = b_by_rows ? e / TSN : e % TSK, bc = b_by_rows ? e % TSN : e / TSK;                       \
            const long gq = t + bp, gc = col0 + bc;                                                                \
            bs[bp][bc] = (gq < k && gc < n) ? b[gq * sb0 + gc * sb1] : (T)0;                                       \
        }                                                                                                          \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                              \
        const int last = k - t < TSK ? (int)(k - t) : TSK;                                                         \
        for (int p = 0; p < last; ++p) {                                                                           \
            T ar[WPT], br[WPT];                                                                                    \
            for (int i = 0; i < WPT; ++i)                                                                          \
                ar[i] = as[p][tr + i * RTS];                                                                       \
            for (int j = 0; j < WPT; ++j)                                                                          \
                br[j] = bs[p][tc + j * RTS];                                                                       \
            for (int i = 0; i < WPT; ++i)                                                                          \
                for (int j = 0; j < WPT; ++j)                                                                      \
                    acc[i][j] += ar[i] * br[j];                                                                    \
        }                                                                                                          \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                              \
    }                                                                                                              \
    for (int i = 0; i < WPT; ++i)                                                                                  \
        for (int j = 0; j < WPT; ++j) {                                                                            \
            const long row = row0 + tr + i * RTS, col = col0 + tc + j * RTS;                                       \
            if (row < m && col < n) {                                                                              \
                __global T* out = c + row * sc0 + col * sc1;                                                       \
                const T s = acc[i][j];                                                                             \
                *out = beta == (T)0 ? alpha * s : alpha * s + beta * *out;                                         \
            }                                                                                                      \
        }                                                                                                          \
}
GEMM(gemm_float, float)
#ifdef SNICHOLLS_GPU_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
GEMM(gemm_double, double)
#endif
)CLC";

    inline void check(cl_int code, const char* what)
    {
        if (code != CL_SUCCESS)
            throw opencl_exception(what, code);
    }

    // A device buffer released when it leaves scope
    struct buffer {
        cl_mem mem = nullptr;
        buffer(cl_context context, cl_mem_flags flags, std::size_t bytes, void* host)
        {
            cl_int code = CL_SUCCESS;
            if (bytes == 0)   // an empty span: a one-byte buffer the kernel never reads
                flags &= ~cl_mem_flags(CL_MEM_COPY_HOST_PTR);
            mem = clCreateBuffer(context, flags, bytes == 0 ? 1 : bytes, bytes == 0 ? nullptr : host, &code);
            check(code, "clCreateBuffer");
        }
        buffer(const buffer&) = delete;
        buffer& operator=(const buffer&) = delete;
        ~buffer()
        {
            if (mem)
                clReleaseMemObject(mem);
        }
    };

} // namespace detail

/// \brief One OpenCL device with its context, queue and compiled kernels.
///
/// Create it once (compiling the kernels takes milliseconds) and hand `opencl_backend{&context}` to core's free
/// functions. Neither copyable nor movable: backends point at it.
class opencl_context {
public:
    /// \brief Finds a device, creates its context and queue, and compiles the kernels.
    /// \param choice A GPU only, or any device.
    /// \return The context, or why there is none.
    [[nodiscard]] static result<std::unique_ptr<opencl_context>, opencl_failure> create(device_choice choice = device_choice::any)
    {
        cl_uint platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &platforms) != CL_SUCCESS || platforms == 0)
            return err(opencl_failure{opencl_error::no_platform, CL_SUCCESS, {}});
        std::vector<cl_platform_id> ids(platforms);
        clGetPlatformIDs(platforms, ids.data(), nullptr);
        cl_device_id device = nullptr;
        for (const cl_device_type type : {cl_device_type(CL_DEVICE_TYPE_GPU), cl_device_type(CL_DEVICE_TYPE_ALL)}) {
            if (type == CL_DEVICE_TYPE_ALL && choice == device_choice::gpu)
                break;
            for (const cl_platform_id p : ids)
                if (!device && clGetDeviceIDs(p, type, 1, &device, nullptr) != CL_SUCCESS)
                    device = nullptr;
            if (device)
                break;
        }
        if (!device)
            return err(opencl_failure{opencl_error::no_device, CL_SUCCESS, {}});
        std::unique_ptr<opencl_context> c(new opencl_context(device));
        cl_int code = CL_SUCCESS;
        c->context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &code);
        if (code != CL_SUCCESS)
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        c->queue_ = clCreateCommandQueue(c->context_, device, 0, &code);
        if (code != CL_SUCCESS)
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        const char* source = detail::gemm_source;   // macOS takes const char**, others const char* const*
        c->program_ = clCreateProgramWithSource(c->context_, 1, &source, nullptr, &code);
        if (code != CL_SUCCESS)
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        const char* options = c->fp64_ ? "-DSNICHOLLS_GPU_FP64" : "";
        if (const cl_int built = clBuildProgram(c->program_, 1, &device, options, nullptr, nullptr); built != CL_SUCCESS) {
            std::size_t size = 0;
            clGetProgramBuildInfo(c->program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            clGetProgramBuildInfo(c->program_, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            return err(opencl_failure{opencl_error::build_failed, built, std::move(log)});
        }
        c->gemm_float_ = clCreateKernel(c->program_, "gemm_float", &code);
        if (code != CL_SUCCESS)
            return err(opencl_failure{opencl_error::call_failed, code, {}});
        if (c->fp64_) {
            c->gemm_double_ = clCreateKernel(c->program_, "gemm_double", &code);
            if (code != CL_SUCCESS)
                return err(opencl_failure{opencl_error::call_failed, code, {}});
        }
        return ok(std::move(c));
    }

    /// \brief Not copyable: backends point at it.
    opencl_context(const opencl_context&) = delete;
    /// \brief Not assignable. \return Nothing; deleted.
    opencl_context& operator=(const opencl_context&) = delete;
    /// \brief Releases the kernels, the program, the queue and the context.
    ~opencl_context() { release(); }

    /// \brief The device's name, as the driver reports it. \return The name.
    [[nodiscard]] std::string device_name() const { return info(CL_DEVICE_NAME); }
    /// \brief Whether the device is a GPU. \return True for a GPU.
    [[nodiscard]] bool is_gpu() const noexcept { return gpu_; }
    /// \brief Whether `double` products run on the device (`cl_khr_fp64`), not on the host. \return True if they do.
    [[nodiscard]] bool has_fp64() const noexcept { return fp64_; }

    /// \brief `C = alpha A B + beta C` on the device. Called by `opencl_backend::gemm`.
    /// \param alpha The scale of the product.
    /// \param a `m x k`.
    /// \param b `k x n`.
    /// \param beta The scale of `C`.
    /// \param c `m x n`; may not overlap `a` or `b`.
    /// \throws opencl_exception if an OpenCL call fails.
    template <typename T>
    void gemm(T alpha, linalg::matrix_view<const T> a, linalg::matrix_view<const T> b, T beta, linalg::matrix_view<T> c) const
    {
        const std::size_t m = c.extent(0), n = c.extent(1), k = a.extent(1);
        if (m == 0 || n == 0)
            return;
        const std::size_t sa = a.mapping().required_span_size(), sb = b.mapping().required_span_size(),
                          sc = c.mapping().required_span_size();
        const cl_mem_flags in = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
        // An empty inner dimension still writes alpha * 0 (+ beta c): the kernel's loop runs no times
        detail::buffer da(context_, in, sa * sizeof(T), const_cast<T*>(a.data_handle()));
        detail::buffer db(context_, in, sb * sizeof(T), const_cast<T*>(b.data_handle()));
        // C goes up only if the kernel reads it (beta != 0) or its span has gaps the copy back must keep
        const bool upload = beta != T(0) || !c.mapping().is_exhaustive();
        detail::buffer dc(context_, upload ? CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR : CL_MEM_WRITE_ONLY, sc * sizeof(T),
                          upload ? c.data_handle() : nullptr);
        const cl_long args[] = {cl_long(m), cl_long(n), cl_long(k), cl_long(a.stride(0)), cl_long(a.stride(1)), cl_long(b.stride(0)),
                                cl_long(b.stride(1)), cl_long(c.stride(0)), cl_long(c.stride(1))};
        const std::lock_guard lock(mutex_);
        cl_kernel kernel = std::is_same_v<T, float> ? gemm_float_ : gemm_double_;
        cl_uint i = 0;
        auto set = [&](std::size_t size, const void* value) { detail::check(clSetKernelArg(kernel, i++, size, value), "clSetKernelArg"); };
        set(sizeof(cl_long), &args[0]), set(sizeof(cl_long), &args[1]), set(sizeof(cl_long), &args[2]), set(sizeof(T), &alpha);
        set(sizeof(cl_mem), &da.mem), set(sizeof(cl_long), &args[3]), set(sizeof(cl_long), &args[4]);
        set(sizeof(cl_mem), &db.mem), set(sizeof(cl_long), &args[5]), set(sizeof(cl_long), &args[6]);
        set(sizeof(T), &beta), set(sizeof(cl_mem), &dc.mem), set(sizeof(cl_long), &args[7]), set(sizeof(cl_long), &args[8]);
        const std::size_t global[2] = {(n + 63) / 64 * 16, (m + 63) / 64 * 16}, local[2] = {16, 16};   // 64 x 64 per group
        detail::check(clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, global, local, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        detail::check(clEnqueueReadBuffer(queue_, dc.mem, CL_TRUE, 0, sc * sizeof(T), c.data_handle(), 0, nullptr, nullptr),
                      "clEnqueueReadBuffer");
    }

private:
    explicit opencl_context(cl_device_id device) : device_(device)
    {
        cl_device_type type = 0;
        clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
        gpu_ = (type & CL_DEVICE_TYPE_GPU) != 0;
        fp64_ = info(CL_DEVICE_EXTENSIONS).find("cl_khr_fp64") != std::string::npos;
    }

    void release() noexcept
    {
        if (gemm_float_)
            clReleaseKernel(gemm_float_);
        if (gemm_double_)
            clReleaseKernel(gemm_double_);
        if (program_)
            clReleaseProgram(program_);
        if (queue_)
            clReleaseCommandQueue(queue_);
        if (context_)
            clReleaseContext(context_);
    }

    [[nodiscard]] std::string info(cl_device_info what) const
    {
        std::size_t size = 0;
        clGetDeviceInfo(device_, what, 0, nullptr, &size);
        std::string s(size, '\0');
        clGetDeviceInfo(device_, what, size, s.data(), nullptr);
        while (!s.empty() && s.back() == '\0')
            s.pop_back();
        return s;
    }

    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel gemm_float_ = nullptr;
    cl_kernel gemm_double_ = nullptr;
    bool gpu_ = false;
    bool fp64_ = false;
    mutable std::mutex mutex_;
};

/// \brief The `linalg::blas_backend` on an `opencl_context`: `gemm` and `gemv` on the device, the rest on the host.
///
/// Trivially copyable (one pointer), as the concept requires; pass it by value to core's free functions. `float`
/// runs on the device; `double` too when the device has `cl_khr_fp64`; any other scalar type, on the host.
struct opencl_backend {
    const opencl_context* context = nullptr;   ///< The device; must outlive the backend.

    /// \brief `x · y`, on the host (core's reference). \param x A vector. \param y Its partner. \return The dot product.
    template <linalg::scalar T>
    [[nodiscard]] T dot(linalg::vector_view<const T> x, linalg::vector_view<const T> y) const
    {
        return linalg::reference.dot(x, y);
    }

    /// \brief `y += alpha x`, on the host. \param alpha The scale. \param x The input. \param y The output.
    template <linalg::scalar T>
    void axpy(T alpha, linalg::vector_view<const T> x, linalg::vector_view<T> y) const
    {
        linalg::reference.axpy(alpha, x, y);
    }

    /// \brief `y = alpha A x + beta y`, on the device as a product with one column.
    /// \param alpha The scale. \param a `m x k`. \param x `k`. \param beta The scale of `y`. \param y `m`.
    /// \throws opencl_exception if an OpenCL call fails.
    template <linalg::scalar T>
    void gemv(T alpha, linalg::matrix_view<const T> a, linalg::vector_view<const T> x, T beta, linalg::vector_view<T> y) const
    {
        if (!on_device<T>()) {
            linalg::reference.gemv(alpha, a, x, beta, y);
            return;
        }
        context->gemm(alpha, a, linalg::as_column(x), beta, linalg::as_column(y));
    }

    /// \brief `C = alpha A B + beta C`, on the device.
    /// \param alpha The scale. \param a `m x k`. \param b `k x n`. \param beta The scale of `C`. \param c `m x n`.
    /// \throws opencl_exception if an OpenCL call fails.
    template <linalg::scalar T>
    void gemm(T alpha, linalg::matrix_view<const T> a, linalg::matrix_view<const T> b, T beta, linalg::matrix_view<T> c) const
    {
        if (on_device<T>())
            context->gemm(alpha, a, b, beta, c);
        else
            linalg::reference.gemm(alpha, a, b, beta, c);
    }

    /// \brief `C = alpha A Aᵀ + beta C` in one triangle, on the host.
    /// \param alpha The scale. \param a `n x k`. \param beta The scale of `C`. \param c `n x n`. \param t The triangle.
    template <linalg::scalar T>
    void syrk(T alpha, linalg::matrix_view<const T> a, T beta, linalg::matrix_view<T> c, linalg::triangle t) const
    {
        linalg::reference.syrk(alpha, a, beta, c, t);
    }

    /// \brief Solves `T X = B` in place for a triangular `T`, on the host.
    /// \param t The triangular matrix. \param b The right-hand sides, overwritten. \param which Its triangle.
    /// \param diag Its diagonal. \return Failure for a singular `T`.
    template <linalg::scalar T>
    [[nodiscard]] result<void, linalg::error> triangular_solve(linalg::matrix_view<const T> t, linalg::matrix_view<T> b,
                                                               linalg::triangle which, linalg::diagonal diag) const
    {
        return linalg::reference.triangular_solve(t, b, which, diag);
    }

private:
    // float always runs on the device, double when it has cl_khr_fp64, any other scalar on the host
    template <typename T>
    [[nodiscard]] bool on_device() const noexcept
    {
        return context != nullptr && (std::is_same_v<T, float> || (std::is_same_v<T, double> && context->has_fp64()));
    }
};

static_assert(linalg::blas_backend<opencl_backend, float> && linalg::blas_backend<opencl_backend, double>);

} // namespace snicholls::gpu

#endif /* snicholls_gpu_opencl_hpp */
