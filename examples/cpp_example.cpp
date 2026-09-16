// Using the C++ core directly: the illustrative example of Kusumo et al. (2020).
// y = p * d0^2 + d1 must lie in [0.2, 0.75] for the nominal parameter p = 1.

#include <cstdio>

#include "nested_sampler/nested_sampler.hpp"

int main() {
  ns::Options options;
  options.num_live = 200;
  options.num_prop = 16;
  options.display = 1;

  const std::vector<double> lb(2, -1.0), ub(2, 1.0);
  ns::NestedSampler sampler(lb, ub, options);

  ns::Evaluator evaluate = [](const std::vector<std::vector<double>>& X) {
    std::vector<std::vector<double>> G(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) {
      const double y = 1.0 * X[i][0] * X[i][0] + X[i][1];
      G[i] = {0.2 - y, y - 0.75};
    }
    return G;
  };

  const ns::Result r = sampler.run(evaluate);
  std::printf("status %d, %zu iterations, %zu evaluations, %zu live, %zu dead, %zu discarded\n",
              static_cast<int>(r.status), r.stats.iterations, r.stats.evaluations,
              r.live.size(), r.dead.size(), r.discarded.size());
  for (std::size_t i = 0; i < 3 && i < r.live.size(); ++i)
    std::printf("  live[%zu] = (%.6f, %.6f) value %.3e\n", i, r.live[i].x[0], r.live[i].x[1], r.live[i].value);
  return r.status == ns::NORMAL ? 0 : 1;
}
