#ifndef NESTED_SAMPLER_LINALG_HPP
#define NESTED_SAMPLER_LINALG_HPP

#include <cmath>
#include <cstddef>
#include <vector>

// Small dense routines for n x n matrices stored row-major in std::vector.

namespace ns {
namespace linalg {

// Euclidean norm with the two-accumulator summation used by Armadillo.
inline double norm2(const double* v, std::size_t n) {
  double acc1 = 0.0, acc2 = 0.0;
  std::size_t i = 0, j = 1;
  for (; j < n; i += 2, j += 2) {
    acc1 += v[i] * v[i];
    acc2 += v[j] * v[j];
  }
  if (i < n) acc1 += v[i] * v[i];
  return std::sqrt(acc1 + acc2);
}

// Lower Cholesky factor of a symmetric matrix; false if not positive definite.
inline bool cholesky_lower(const std::vector<double>& a, std::size_t n, std::vector<double>& l) {
  l.assign(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    double d = a[j * n + j];
    for (std::size_t k = 0; k < j; ++k) d -= l[j * n + k] * l[j * n + k];
    if (!(d > 0.0)) return false;
    d = std::sqrt(d);
    l[j * n + j] = d;
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = a[i * n + j];
      for (std::size_t k = 0; k < j; ++k) s -= l[i * n + k] * l[j * n + k];
      l[i * n + j] = s / d;
    }
  }
  return true;
}

// Solve L x = b for lower-triangular L by forward substitution.
inline void solve_lower(const std::vector<double>& l, std::size_t n, const double* b, double* x) {
  for (std::size_t i = 0; i < n; ++i) {
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= l[i * n + k] * x[k];
    x[i] = s / l[i * n + i];
  }
}

// y = A v
inline void mat_vec(const std::vector<double>& a, std::size_t n, const double* v, double* y) {
  for (std::size_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (std::size_t k = 0; k < n; ++k) s += a[i * n + k] * v[k];
    y[i] = s;
  }
}

}  // namespace linalg
}  // namespace ns

#endif
