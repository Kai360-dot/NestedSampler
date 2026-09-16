#ifndef NESTED_SAMPLER_RNG_HPP
#define NESTED_SAMPLER_RNG_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "sobol.hpp"

namespace ns {

// Pseudo-random stream reproducing what MAGNUS obtains from Armadillo on Linux:
// a std::mt19937_64 engine (fully specified by the C++ standard, hence identical
// on every platform) combined with the libstdc++ implementations of
// uniform and normal deviates, which are written out here so that MSVC and
// libc++ builds produce the same numbers.
class Rng {
public:
  explicit Rng(std::uint64_t seed = 5489u) : engine_(seed) {}

  void seed(std::uint64_t s) { engine_.seed(s); }
  void discard(unsigned long long n) { engine_.discard(n); }

  // arma::randu<double>(): 64-bit draw scaled by 2^-64 (can return exactly 1.0).
  double uniform() {
    return uint64_to_double(engine_()) * (1.0 / 18446744073709551616.0);
  }

  // arma::randn(n): n standard normals from one fresh std::normal_distribution
  // (libstdc++ Marsaglia polar method, the spare deviate of the last pair is
  // dropped when n is odd).
  void normals(double* out, std::size_t n) {
    bool spare_available = false;
    double spare = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (spare_available) {
        spare_available = false;
        out[i] = spare;
        continue;
      }
      double x, y, r2;
      do {
        x = 2.0 * canonical() - 1.0;
        y = 2.0 * canonical() - 1.0;
        r2 = x * x + y * y;
      } while (r2 > 1.0 || r2 == 0.0);
      const double mult = std::sqrt(-2 * std::log(r2) / r2);
      spare = x * mult;
      spare_available = true;
      out[i] = y * mult;
    }
  }

private:
  // std::generate_canonical<double, 53>(std::mt19937_64&) as in libstdc++.
  double canonical() {
    double u = uint64_to_double(engine_()) / 18446744073709551616.0;
    if (u >= 1.0) u = std::nextafter(1.0, 0.0);
    return u;
  }

  std::mt19937_64 engine_;
};

}  // namespace ns

#endif
