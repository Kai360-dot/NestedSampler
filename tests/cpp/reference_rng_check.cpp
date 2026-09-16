// Checks that the random streams of the port are bit-identical to those of the
// reference build: Boost's 64-bit Sobol engine and, on libstdc++, the standard
// normal and canonical uniform distributions used by Armadillo.

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <cstdio>
#include <random>

#include "nested_sampler/rng.hpp"
#include "nested_sampler/sobol.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    ++failures;
    std::printf("FAILED: %s\n", what);
  }
}

void check_sobol(std::size_t dim, std::size_t points) {
  typedef boost::random::sobol_engine<boost::uint_least64_t, 64u> sobol64;
  sobol64 engine(dim);
  boost::variate_generator<sobol64, boost::uniform_01<double>> gen(engine, boost::uniform_01<double>());
  gen.engine().seed(0);

  ns::Sobol mine(dim);
  mine.seed(0);
  for (std::size_t k = 0; k < points * dim; ++k) {
    if (gen() != mine.next()) {
      std::printf("Sobol mismatch: dim %zu, coordinate %zu\n", dim, k);
      ++failures;
      return;
    }
  }
}

void check_rng() {
#ifdef __GLIBCXX__
  const unsigned long long seed = 5489u;
  std::mt19937_64 engine(seed);
  ns::Rng mine(seed);
  double buf[7];
  for (int round = 0; round < 2000; ++round) {
    // arma::randu(): 64-bit draw scaled by 1 / mt19937_64::max()
    const double ref_u = static_cast<double>(engine()) * (1.0 / static_cast<double>(std::mt19937_64::max()));
    check(ref_u == mine.uniform(), "uniform");

    // arma::randn(n): fresh std::normal_distribution per call
    const std::size_t n = 1 + round % 7;
    std::normal_distribution<double> normal;
    mine.normals(buf, n);
    for (std::size_t i = 0; i < n; ++i) check(normal(engine) == buf[i], "normal");
  }
#else
  std::printf("standard library is not libstdc++: normal/uniform check skipped\n");
#endif
}

}  // namespace

int main() {
  for (std::size_t dim = 1; dim <= 12; ++dim) check_sobol(dim, 4000);
  check_sobol(100, 200);
  check_rng();
  if (failures == 0) std::printf("reference_rng_check: all streams identical\n");
  return failures == 0 ? 0 : 1;
}
