#ifndef NESTED_SAMPLER_CRITERION_HPP
#define NESTED_SAMPLER_CRITERION_HPP

#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>

namespace ns {

// Feasibility criterion of one design point, computed from the constraint
// values g[s * ng + j] (scenario s, constraint j; feasible iff all g <= 0)
// exactly as MAGNUS's FFNSamp::_feasval does.
struct Criterion {
  double prob;  // probability of feasibility across the scenarios
  double var;   // value-at-risk of the maximal constraint value
  double cvar;  // conditional value-at-risk of the maximal constraint value
};

inline Criterion feasibility_criterion(const double* g, std::size_t ns, std::size_t ng,
                                       const std::vector<double>& weights, double conf) {
  Criterion c;
  c.prob = 0.0;
  c.var = 0.0;
  c.cvar = 0.0;

  // Single scenario: the maximal constraint value itself.
  if (ns <= 1) {
    const double m = *std::max_element(g, g + ng);
    c.prob = (m <= 0.0) ? 1.0 : 0.0;
    c.var = c.cvar = m;
    return c;
  }

  // Scenarios ordered by largest violation first (key is the negated maximum).
  std::multimap<double, double> ordered;
  for (std::size_t s = 0; s < ns; ++s) {
    const double m = *std::max_element(g + s * ng, g + (s + 1) * ng);
    ordered.insert(std::make_pair(-m, weights[s]));
  }

  // Probability of feasibility.
  c.prob = 1.0;
  for (std::multimap<double, double>::const_iterator it = ordered.begin(); it != ordered.end(); ++it) {
    if (it->first >= 0.0) break;
    c.prob -= it->second;
  }

  // Value-at-risk.
  double mass = 0.0, var = 0.0;
  for (std::multimap<double, double>::const_iterator it = ordered.begin(); it != ordered.end(); ++it) {
    var = it->first;
    if (mass + it->second > conf) break;
    mass += it->second;
  }
  c.var = c.cvar = -var;

  // Conditional value-at-risk.
  for (std::multimap<double, double>::const_iterator it = ordered.begin(); it != ordered.end(); ++it) {
    if (it->first > var) break;
    c.cvar += (var - it->first) * it->second / conf;
  }
  return c;
}

}  // namespace ns

#endif
