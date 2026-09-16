#ifndef NESTED_SAMPLER_SOBOL_HPP
#define NESTED_SAMPLER_SOBOL_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "sobol_table.hpp"

namespace ns {

// Nearest double to a 64-bit unsigned integer, written out so that the result
// does not depend on how a particular compiler implements the conversion.
inline double uint64_to_double(std::uint64_t x) {
  return static_cast<double>(x >> 11) * 2048.0 + static_cast<double>(x & 2047u);
}

// 64-bit Sobol sequence, bit-identical to boost::random::sobol_engine<uint64_t, 64>
// which MAGNUS uses: Joe-Kuo direction numbers, Gray-code ordering, and the
// all-zero point skipped, so the first point is 0.5 in every dimension.
class Sobol {
public:
  explicit Sobol(std::size_t dimension) : dim_(dimension), count_(0), elem_(0) {
    if (dimension == 0 || dimension > sobol_table::max_dimension)
      throw std::invalid_argument("Sobol: dimension must be between 1 and " +
                                  std::to_string(sobol_table::max_dimension));
    init_lattice();
    seed(0);
  }

  std::size_t dimension() const { return dim_; }

  // Restart the sequence at point number index (0 = first point).
  void seed(std::uint64_t index = 0) {
    std::uint64_t code = index + 1;
    if (code == 0) throw std::range_error("Sobol: seed out of range");
    code ^= (code >> 1);
    state_.assign(dim_, 0);
    for (unsigned bit = 0; code != 0; ++bit, code >>= 1)
      if (code & 1u) flip(bit);
    count_ = index;
    elem_ = 0;
  }

  // Next coordinate as a 64-bit integer; dimension() calls make up one point.
  std::uint64_t next_integer() {
    if (elem_ == dim_) {
      std::uint64_t next = count_ + 1;
      if (next == 0) throw std::range_error("Sobol: sequence exhausted");
      flip(lowest_zero_bit(next));
      count_ = next;
      elem_ = 0;
    }
    return state_[elem_++];
  }

  // Next coordinate in [0, 1), with the semantics of boost::uniform_01 over the
  // engine: values that round up to 1.0 are skipped.
  double next() {
    for (;;) {
      double u = uint64_to_double(next_integer()) * (1.0 / 18446744073709551616.0);
      if (u < 1.0) return u;
    }
  }

private:
  static unsigned lowest_zero_bit(std::uint64_t n) {
    unsigned r = 0;
    while (n & 1u) { n >>= 1; ++r; }
    return r;
  }

  static unsigned integer_log2(unsigned v) {
    unsigned r = 0;
    while (v >>= 1) ++r;
    return r;
  }

  // lattice_[bit * dim_ + d] is direction number v_bit of dimension d.
  void init_lattice() {
    const unsigned bits = 64;
    lattice_.assign(bits * dim_, 0);
    for (unsigned k = 0; k != bits; ++k) lattice_[dim_ * k] = 1;
    for (std::size_t d = 1; d < dim_; ++d) {
      const unsigned poly = sobol_table::polynomial[d - 1];
      const unsigned degree = integer_log2(poly);
      for (unsigned k = 0; k != degree; ++k)
        lattice_[dim_ * k + d] = sobol_table::minit[sobol_table::max_degree * (d - 1) + k];
      for (unsigned j = degree; j < bits; ++j) {
        unsigned p = poly;
        std::uint64_t v = lattice_[dim_ * (j - degree) + d];
        for (unsigned k = 0; k != degree; ++k, p >>= 1) {
          const unsigned rem = degree - k;
          v ^= (static_cast<std::uint64_t>(p & 1u) * lattice_[dim_ * (j - rem) + d]) << rem;
        }
        lattice_[dim_ * j + d] = v;
      }
    }
    for (unsigned j = 0; j + 1 < bits; ++j)
      for (std::size_t d = 0; d != dim_; ++d)
        lattice_[dim_ * j + d] <<= (bits - 1 - j);
  }

  void flip(unsigned bit) {
    for (std::size_t d = 0; d != dim_; ++d) state_[d] ^= lattice_[dim_ * bit + d];
  }

  std::size_t dim_;
  std::vector<std::uint64_t> lattice_;
  std::vector<std::uint64_t> state_;
  std::uint64_t count_;
  std::size_t elem_;
};

}  // namespace ns

#endif
