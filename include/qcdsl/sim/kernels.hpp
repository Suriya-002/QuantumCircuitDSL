#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define QCDSL_X86 1
#include <immintrin.h>
#else
#define QCDSL_X86 0
#endif

// NB: no #include <omp.h>. The `#pragma omp` directives below are handled by
// the compiler; the header only declares the OpenMP *runtime* API, which this
// file never calls. Including it needlessly breaks any tool that replays these
// flags through a compiler without an OpenMP header on its include path.

namespace qcdsl {

template <typename Real>
using Matrix2 = std::array<std::complex<Real>, 4>;

/// Which gate kernel to run.
enum class Kernel : std::uint8_t {
  Scalar,    ///< the portable reference. Always correct, always available.
  Simd,      ///< AVX-512, single-threaded.
  Parallel,  ///< AVX-512 across OpenMP threads.
  Auto,  ///< Simd or Parallel when they are legal and worth it, else Scalar.
};

namespace kernel {

/// WHY Auto DOES NOT USE THREADS.
///
/// Measured, 8 threads, 100 single-qubit gates, best of 3:
///
///     qubits   amplitudes    AVX-512   +OpenMP    GB/s
///         14         16 K     1.48 ms   8.72 ms    35
///         18        262 K    18.2  ms  30.6  ms    46
///         20        1.0 M     112  ms  91.8  ms    37   <-- the only win
///         22        4.2 M     643  ms   749  ms    21   <-- bandwidth wall
///         24       16.8 M    2253  ms  2782  ms    24
///
/// Threads pay for themselves between two walls and nowhere else. Below roughly
/// 2^20 amplitudes the fork-join costs more than the gate does. Above it the
/// state vector has left the last-level cache -- watch the achieved bandwidth
/// collapse from 46 GB/s to 21 -- and the kernel is waiting on DRAM. More
/// threads then mean more contention for one memory controller, not more
/// throughput.
///
/// The window is one doubling wide and it moves with the cache size and the
/// channel count of whatever machine you are on. A default that guesses wrong
/// most of the time is worse than a default that does not guess: Auto takes the
/// serial vector kernel, and Kernel::Parallel is there for anyone whose machine
/// has the bandwidth headroom to use it.
///
/// The real fix is not more threads, it is fewer sweeps. Each gate currently
/// streams the entire state vector through the CPU once. Applying several gates
/// to one cache-resident tile before moving to the next divides the memory
/// traffic by the number of gates fused, and turns a bandwidth-bound kernel
/// back into a compute-bound one. That is the planned next kernel, and it is
/// the point at which threads become worth having again.

/// Checked once, at first call. Compiling AVX-512 in is not the same as being
/// allowed to execute it: a binary built with -mavx512f dies with SIGILL on a
/// machine without it, and CI runners are not guaranteed to have it. So the
/// AVX-512 path is compiled unconditionally via a target attribute and gated
/// here at run time.
inline bool has_avx512() noexcept {
#if QCDSL_X86
  // GCC declares __builtin_cpu_supports as returning int; clang declares it as
  // returning bool. The cast is required under GCC and redundant under clang,
  // and clang-tidy only ever sees the latter.
  // NOLINTBEGIN(readability-redundant-casting)
  static const bool supported =
      static_cast<bool>(__builtin_cpu_supports("avx512f"));
  // NOLINTEND(readability-redundant-casting)
  return supported;
#else
  return false;
#endif
}

/// The reference kernel. Every other kernel is checked against this one.
///
/// Pairs are enumerated by inserting a zero bit at position `t`: the bits of k
/// above t shift up one place, the bits below stay put. No branch in the body
/// -- which is exactly what lets the vector version below load contiguously.
template <typename Real>
void apply_1q_scalar(std::complex<Real>* amps, std::size_t dim,
                     const Matrix2<Real>& u, std::size_t t) noexcept {
  using C = std::complex<Real>;
  const std::size_t bit = std::size_t(1) << t;
  const std::size_t low = bit - 1;
  const std::size_t half = dim >> 1;

  for (std::size_t k = 0; k < half; ++k) {
    const std::size_t i = ((k >> t) << (t + 1)) | (k & low);
    const std::size_t j = i | bit;
    const C a = amps[i];
    const C b = amps[j];
    amps[i] = u[0] * a + u[1] * b;
    amps[j] = u[2] * a + u[3] * b;
  }
}

#if QCDSL_X86

/// (re + i*im) * a, for four complex doubles at once.
///
/// A __m512d is [re0 im0 re1 im1 re2 im2 re3 im3]. Swapping each adjacent pair
/// gives [im0 re0 ...], and FMADDSUB subtracts on the even lanes and adds on
/// the odd ones -- which is precisely the complex product:
///     even: re*a.re - im*a.im
///     odd:  re*a.im + im*a.re
__attribute__((target("avx512f"))) inline __m512d cmul(__m512d a, __m512d re,
                                                       __m512d im) noexcept {
  const __m512d swapped = _mm512_permute_pd(a, 0x55);
  const __m512d cross = _mm512_mul_pd(im, swapped);
  return _mm512_fmaddsub_pd(re, a, cross);
}

/// AVX-512 kernel. REQUIRES t >= 2.
///
/// Amplitudes whose bit t is 0 come in contiguous runs of 2^t. For t >= 2 a run
/// is at least four complex doubles -- one full 512-bit register -- so the "0"
/// partners and the "1" partners can each be loaded as a straight vector, with
/// no gather and no shuffle. For t = 0 and t = 1 the partners interleave inside
/// a register and the caller falls back to the scalar kernel; that is 2 of n
/// targets, and it is stated rather than hidden.
__attribute__((target("avx512f"))) inline void apply_1q_avx512(
    std::complex<double>* amps, std::size_t dim, const Matrix2<double>& u,
    std::size_t t) noexcept {
  auto* p = reinterpret_cast<double*>(amps);
  const std::size_t bit = std::size_t(1) << t;
  const std::size_t blocks = dim >> (t + 1);

  const __m512d u00r = _mm512_set1_pd(u[0].real());
  const __m512d u00i = _mm512_set1_pd(u[0].imag());
  const __m512d u01r = _mm512_set1_pd(u[1].real());
  const __m512d u01i = _mm512_set1_pd(u[1].imag());
  const __m512d u10r = _mm512_set1_pd(u[2].real());
  const __m512d u10i = _mm512_set1_pd(u[2].imag());
  const __m512d u11r = _mm512_set1_pd(u[3].real());
  const __m512d u11i = _mm512_set1_pd(u[3].imag());

  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t base = b * (bit << 1);
    for (std::size_t k = 0; k < bit; k += 4) {
      const std::size_t i = base + k;
      const std::size_t j = i + bit;
      const __m512d va = _mm512_loadu_pd(p + 2 * i);
      const __m512d vb = _mm512_loadu_pd(p + 2 * j);
      _mm512_storeu_pd(
          p + 2 * i, _mm512_add_pd(cmul(va, u00r, u00i), cmul(vb, u01r, u01i)));
      _mm512_storeu_pd(
          p + 2 * j, _mm512_add_pd(cmul(va, u10r, u10i), cmul(vb, u11r, u11i)));
    }
  }
}

/// The same kernel across OpenMP threads, parallelised over a FLAT chunk index.
///
/// The obvious thing -- `#pragma omp parallel for` on the block loop -- is
/// wrong, and measurably so. There are `dim >> (t+1)` blocks, and that count
/// collapses as the target qubit rises: at t = n-1 there is exactly ONE block.
/// OpenMP forks its threads, hands every iteration to one of them, and joins.
/// All of the overhead, none of the parallelism. Measured on 8 threads, that
/// version was 1.2x to 5x SLOWER than the serial vector kernel at every size.
///
/// However the outer/inner split falls, the total work is always dim/8 vector
/// chunks. Iterating that flat count and recovering (block, offset) from it
/// keeps every thread fed for every target qubit.
__attribute__((target("avx512f"))) inline void apply_1q_avx512_omp(
    std::complex<double>* amps, std::size_t dim, const Matrix2<double>& u,
    std::size_t t) noexcept {
  auto* p = reinterpret_cast<double*>(amps);
  const std::size_t bit = std::size_t(1) << t;
  const std::size_t blocks = dim >> (t + 1);

  // simd_applies() guarantees t >= 2, so chunks_per_block = bit/4 >= 1 and both
  // divisions below are by a power of two: a shift and a mask.
  const std::size_t shift = t - 2;  // log2(vector chunks per block)
  const std::size_t mask = (std::size_t(1) << shift) - 1;
  const auto chunks = static_cast<std::ptrdiff_t>(blocks << shift);

  const __m512d u00r = _mm512_set1_pd(u[0].real());
  const __m512d u00i = _mm512_set1_pd(u[0].imag());
  const __m512d u01r = _mm512_set1_pd(u[1].real());
  const __m512d u01i = _mm512_set1_pd(u[1].imag());
  const __m512d u10r = _mm512_set1_pd(u[2].real());
  const __m512d u10i = _mm512_set1_pd(u[2].imag());
  const __m512d u11r = _mm512_set1_pd(u[3].real());
  const __m512d u11i = _mm512_set1_pd(u[3].imag());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t c = 0; c < chunks; ++c) {
    const auto uc = static_cast<std::size_t>(c);
    const std::size_t b = uc >> shift;
    const std::size_t k = (uc & mask) << 2;
    const std::size_t i = (b << (t + 1)) + k;
    const std::size_t j = i + bit;
    const __m512d va = _mm512_loadu_pd(p + 2 * i);
    const __m512d vb = _mm512_loadu_pd(p + 2 * j);
    _mm512_storeu_pd(p + 2 * i,
                     _mm512_add_pd(cmul(va, u00r, u00i), cmul(vb, u01r, u01i)));
    _mm512_storeu_pd(p + 2 * j,
                     _mm512_add_pd(cmul(va, u10r, u10i), cmul(vb, u11r, u11i)));
  }
}

#endif  // QCDSL_X86

/// True when the vector kernel is both legal on this CPU and applicable to this
/// target. Exposed so the benchmark and the tests can say what they measured.
template <typename Real>
inline bool simd_applies(std::size_t dim, std::size_t t) noexcept {
  if constexpr (!std::is_same_v<Real, double>) {
    return false;  // the vector kernel is double-only for now
  } else {
    return QCDSL_X86 != 0 && has_avx512() && t >= 2 && dim >= 8;
  }
}

/// Dispatch. Falls back to scalar whenever the fast path does not apply, so a
/// caller can always ask for Simd and always get a correct answer.
template <typename Real>
void apply_1q(std::complex<Real>* amps, std::size_t dim, const Matrix2<Real>& u,
              std::size_t t, Kernel which) noexcept {
#if QCDSL_X86
  if constexpr (std::is_same_v<Real, double>) {
    if (which != Kernel::Scalar && simd_applies<Real>(dim, t)) {
      if (which == Kernel::Parallel) {
        apply_1q_avx512_omp(reinterpret_cast<std::complex<double>*>(amps), dim,
                            reinterpret_cast<const Matrix2<double>&>(u), t);
      } else {
        apply_1q_avx512(reinterpret_cast<std::complex<double>*>(amps), dim,
                        reinterpret_cast<const Matrix2<double>&>(u), t);
      }
      return;
    }
  }
#else
  (void)which;
#endif
  apply_1q_scalar(amps, dim, u, t);
}

}  // namespace kernel
}  // namespace qcdsl
