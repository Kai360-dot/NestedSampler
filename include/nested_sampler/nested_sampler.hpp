#ifndef NESTED_SAMPLER_NESTED_SAMPLER_HPP
#define NESTED_SAMPLER_NESTED_SAMPLER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

// Nested sampling for feasibility analysis, a standalone port of mc::NSFEAS
// from Benoit Chachuat's MAGNUS library (feasibility phase only).

namespace ns {

struct Options {
  enum Criterion { VAR = 1, CVAR = 2 };

  Criterion criterion;         // risk measure of the maximal constraint value (FEASCRIT)
  double threshold;            // VaR/CVaR percentile (FEASTHRES)
  std::size_t num_live;        // number of live points (NUMLIVE)
  std::size_t num_prop;        // proposals per iteration (NUMPROP)
  double ell_mag;              // initial ellipsoid magnification (ELLMAG)
  double ell_red;              // magnification reduction exponent (ELLRED)
  std::size_t max_iter;        // iteration limit, 0 = none (MAXITER)
  std::size_t max_err;         // failed evaluation limit, 0 = none (MAXERR)
  double max_time;             // wall-time limit in seconds, 0 = none (MAXCPU)
  std::uint64_t seed;          // seed of the pseudo-random stream (Armadillo default: 5489)
  unsigned long long rng_skip; // pseudo-random draws discarded after seeding
  int display;                 // 0 = silent, 1 = progress table (DISPLEVEL)
  std::size_t display_iter;    // print every display_iter iterations, 0 = every (DISPITER)

  Options()
    : criterion(VAR), threshold(0.1), num_live(256), num_prop(16), ell_mag(0.30), ell_red(0.20),
      max_iter(0), max_err(0), max_time(0.0), seed(5489u), rng_skip(0), display(1), display_iter(25) {}
};

enum Status {
  NORMAL = 0,    // stop level reached: every live point is feasible
  INTERRUPT = 1  // iteration, failure or time limit reached
};

struct Point {
  std::vector<double> x;  // design point
  double value;           // feasibility criterion (VaR or CVaR); feasible iff <= 0
  double prob;            // probability of feasibility across the scenarios
};

struct Stats {
  std::size_t iterations;   // proposal batches processed
  std::size_t evaluations;  // constraint evaluations (points times scenarios)
  std::size_t failures;     // failed point evaluations
  std::size_t feasible;     // feasible points inserted into the live set
  double walltime;          // seconds
};

struct IterationRecord {
  std::size_t iteration;
  double contour;        // largest criterion in the live set after the iteration
  std::size_t feasible;  // running count of feasible points
  std::size_t dead;      // dead points so far
  double factor;         // ellipsoid magnification factor after the iteration
  bool box_sampling;     // proposals came from the bounding box, not the ellipsoid
};

struct Result {
  Status status;
  Stats stats;
  std::vector<Point> live;       // ascending criterion
  std::vector<Point> dead;       // in order of death
  std::vector<Point> discarded;  // rejected proposals, in order of rejection
  std::vector<IterationRecord> history;
};

// Batch constraint evaluator. Receives m points and returns m vectors holding
// the ns * ng constraint values of each point, scenario-major (g[s * ng + j]).
// An empty vector marks a failed evaluation.
typedef std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> Evaluator;

class NestedSampler {
public:
  // lb, ub: box domain. scenario_weights: probability of each parameter
  // scenario, summing to one (empty for a single scenario). MAGNUS uses
  // exactly 1/ns for equal weights and w/sum(w) otherwise; do the same to
  // reproduce its VaR/CVaR percentile boundaries.
  NestedSampler(std::vector<double> lb, std::vector<double> ub, Options options = Options(),
                std::vector<double> scenario_weights = std::vector<double>());

  Result run(const Evaluator& evaluate, std::ostream& os = std::cout) const;

  std::size_t dimension() const { return lb_.size(); }
  std::size_t num_scenarios() const { return weights_.empty() ? 1 : weights_.size(); }
  const Options& options() const { return options_; }

private:
  std::vector<double> lb_, ub_, weights_;
  Options options_;
};

}  // namespace ns

#endif
