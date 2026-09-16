#include "nested_sampler/nested_sampler.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>

#include "nested_sampler/criterion.hpp"
#include "nested_sampler/linalg.hpp"
#include "nested_sampler/rng.hpp"
#include "nested_sampler/sobol.hpp"

namespace ns {

namespace {

typedef std::chrono::steady_clock Clock;
typedef std::multimap<double, Point> LiveSet;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Restores stream formatting on scope exit.
struct FormatGuard {
  explicit FormatGuard(std::ostream& os) : os_(os), flags_(os.flags()), precision_(os.precision()) {}
  ~FormatGuard() { os_.flags(flags_); os_.precision(precision_); }
  std::ostream& os_;
  std::ios::fmtflags flags_;
  std::streamsize precision_;
};

struct Evaluation {
  bool ok;
  double value;
  double prob;
};

// One sampling run; mirrors NSFEAS::sample, _sample_ini, _update_nest,
// _sample_nest, _sample_feas and _terminate of MAGNUS.
class Run {
public:
  Run(const std::vector<double>& lb, const std::vector<double>& ub, const std::vector<double>& weights,
      const Options& opt, const Evaluator& evaluate, std::ostream& os)
    : lb_(lb), ub_(ub), weights_(weights), opt_(opt), evaluate_(evaluate), os_(os),
      nx_(lb.size()), ns_(weights.empty() ? 1 : weights.size()),
      sobol_(nx_), rng_(opt.seed), factor_(opt.ell_mag),
      nest_lb_(lb), nest_ub_(ub), centre_(nx_, 0.0), shape_(nx_ * nx_, 0.0),
      z_(nx_), y_(nx_), iterations_(0), t0_(Clock::now()) {
    rng_.discard(opt.rng_skip);
    stats_.iterations = stats_.evaluations = stats_.failures = stats_.feasible = 0;
    stats_.walltime = 0.0;
  }

  Result execute() {
    int flag = initialise() ? sample_feasibility() : 2;

    Result result;
    result.status = (flag == 1) ? NORMAL : INTERRUPT;
    stats_.iterations = iterations_;
    stats_.walltime = seconds_since(t0_);
    result.stats = stats_;
    for (LiveSet::const_iterator it = live_.begin(); it != live_.end(); ++it)
      result.live.push_back(it->second);
    result.dead = dead_;
    result.discarded = discarded_;
    result.history = history_;
    return result;
  }

private:
  // Fill the live set with Sobol points of the box domain, resampling failures.
  bool initialise() {
    if (opt_.display) os_ << "\n** INITIALIZING LIVE POINTS " << std::flush;

    bool ok = true;
    while (ok && live_.size() < opt_.num_live) {
      std::vector<std::vector<double>> X;
      while (X.size() < opt_.num_live - live_.size()) X.push_back(sample_box());
      const std::vector<Evaluation> ev = evaluate_batch(X);
      for (std::size_t i = 0; i < X.size(); ++i) {
        if (!ev[i].ok) {
          if (opt_.max_err > 0 && stats_.failures >= opt_.max_err) { ok = false; break; }
          continue;
        }
        insert_live(X[i], ev[i]);
      }
      if (opt_.max_time > 0.0 && seconds_since(t0_) >= opt_.max_time) ok = false;
    }

    if (opt_.display) {
      FormatGuard guard(os_);
      os_ << "(" << live_.size() << (ok ? ")" : ") INTERRUPTED AT")
          << std::right << std::fixed << std::setprecision(2)
          << std::setw(10) << seconds_since(t0_) << " SEC" << std::endl;
    }
    return ok;
  }

  // Main loop: 1 = stop level reached, 2 = resource limit reached.
  int sample_feasibility() {
    if (opt_.display) {
      os_ << std::endl
          << std::setw(7) << "Iterate" << std::setw(15) << "Contour" << std::setw(6) << "#Feas"
          << std::setw(9) << "#Dead" << std::setw(15) << "Factor" << std::endl
          << std::setw(7 + 15 + 6 + 9 + 15) << std::setfill('-') << "-" << std::setfill(' ') << std::endl;
    }

    int flag = terminate(0);
    for (iterations_ = 0; !flag; ++iterations_) {
      update_nest();
      if (opt_.display && (opt_.display_iter == 0 || iterations_ % opt_.display_iter == 0))
        display_row(iterations_);

      // Proposals are all generated first, then evaluated as one batch.
      std::vector<std::vector<double>> X;
      bool box = false;
      while (X.size() < opt_.num_prop) sample_nest(X, box);
      const std::vector<Evaluation> ev = evaluate_batch(X);

      // Every proposal of the batch is processed, even once the stop level is
      // reached within the batch (as in MAGNUS).
      for (std::size_t i = 0; i < X.size(); ++i) {
        if (!ev[i].ok) continue;
        if (ev[i].value > worst()) {
          discarded_.push_back(make_point(X[i], ev[i]));
          continue;
        }
        insert_live(X[i], ev[i]);
        const double nest_mass = std::exp(-static_cast<double>(dead_.size()) / static_cast<double>(opt_.num_live));
        kill_worst();
        factor_ = opt_.ell_mag * std::pow(nest_mass, opt_.ell_red);
        flag = terminate(iterations_);
      }
      if (!flag && opt_.max_err > 0 && stats_.failures >= opt_.max_err) flag = 2;

      IterationRecord rec;
      rec.iteration = iterations_;
      rec.contour = worst();
      rec.feasible = stats_.feasible;
      rec.dead = dead_.size();
      rec.factor = factor_;
      rec.box_sampling = box;
      history_.push_back(rec);
    }

    if (opt_.display) display_row(iterations_);
    return flag;
  }

  int terminate(std::size_t iteration) const {
    if (worst() <= 0.0) return 1;
    if (opt_.max_time > 0.0 && seconds_since(t0_) >= opt_.max_time) return 2;
    if (opt_.max_iter > 0 && iteration >= opt_.max_iter) return 2;
    return 0;
  }

  // Bounding box and enclosing ellipsoid of the live points.
  void update_nest() {
    const std::size_t n = nx_;
    const double N = static_cast<double>(live_.size());

    for (std::size_t i = 0; i < n; ++i) {
      nest_lb_[i] = DBL_MAX;
      nest_ub_[i] = -DBL_MAX;
      centre_[i] = 0.0;
    }
    for (LiveSet::const_iterator it = live_.begin(); it != live_.end(); ++it) {
      const std::vector<double>& x = it->second.x;
      for (std::size_t i = 0; i < n; ++i) {
        if (x[i] < nest_lb_[i]) nest_lb_[i] = x[i];
        if (x[i] > nest_ub_[i]) nest_ub_[i] = x[i];
        centre_[i] += x[i];
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      const double corr = (nest_ub_[i] - nest_lb_[i]) * (0.5 * factor_);
      nest_lb_[i] = std::max(nest_lb_[i] - corr, lb_[i]);
      nest_ub_[i] = std::min(nest_ub_[i] + corr, ub_[i]);
      centre_[i] /= N;
    }

    // Covariance normalised by the number of points, as arma::cov(X, 1).
    std::vector<double> cov(n * n, 0.0);
    for (LiveSet::const_iterator it = live_.begin(); it != live_.end(); ++it) {
      const std::vector<double>& x = it->second.x;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          cov[i * n + j] += (x[i] - centre_[i]) * (x[j] - centre_[j]);
    }
    for (std::size_t k = 0; k < n * n; ++k) cov[k] /= N;

    if (!linalg::cholesky_lower(cov, n, shape_)) {
      if (opt_.display) os_ << "Cholesky factorization failed" << std::endl;
      shape_.assign(n * n, 0.0);
      const double c = 0.5 * std::sqrt(static_cast<double>(n));
      for (std::size_t i = 0; i < n; ++i) {
        centre_[i] = 0.5 * (nest_lb_[i] + nest_ub_[i]);
        shape_[i * n + i] = c * (nest_ub_[i] - nest_lb_[i]);
      }
      return;
    }

    // Scale the factor so that the ellipsoid encloses the farthest live point.
    double max_mag = 0.0;
    for (LiveSet::const_iterator it = live_.begin(); it != live_.end(); ++it) {
      const std::vector<double>& x = it->second.x;
      for (std::size_t i = 0; i < n; ++i) z_[i] = x[i] - centre_[i];
      linalg::solve_lower(shape_, n, z_.data(), y_.data());
      max_mag = std::max(max_mag, linalg::norm2(y_.data(), n));
    }
    const double scale = max_mag * (1 + factor_);
    for (std::size_t k = 0; k < n * n; ++k) shape_[k] *= scale;
  }

  // Draw one proposal from the box (Sobol) when it is not larger than the
  // ellipsoid, otherwise uniformly from the ellipsoid; ellipsoid draws that
  // leave the domain are rejected.
  void sample_nest(std::vector<std::vector<double>>& X, bool& box) {
    const std::size_t n = nx_;
    double logvol_box = 0.0, logvol_nest = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double w = nest_ub_[i] - nest_lb_[i];
      logvol_box += std::log(w > DBL_EPSILON ? w : DBL_EPSILON);
      const double s = shape_[i * n + i];
      logvol_nest += std::log(s > DBL_EPSILON ? s : DBL_EPSILON);
    }
    if (logvol_box <= logvol_nest) {
      X.push_back(sample_box());
      box = true;
      return;
    }

    // Uniform point in the unit ball. The radius uniform is drawn before the
    // normals: that is the operand order the reference build uses.
    const double u = rng_.uniform();
    rng_.normals(z_.data(), n);
    double norm = linalg::norm2(z_.data(), n);
    if (norm == 0.0) norm = 1.0;
    const double radius = std::pow(u, 1 / static_cast<double>(n));
    for (std::size_t i = 0; i < n; ++i) z_[i] = (z_[i] / norm) * radius;

    linalg::mat_vec(shape_, n, z_.data(), y_.data());
    std::vector<double> p(n);
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = centre_[i] + y_[i];
      if (p[i] < lb_[i] || p[i] > ub_[i]) return;
    }
    X.push_back(p);
  }

  std::vector<double> sample_box() {
    std::vector<double> p(nx_);
    for (std::size_t i = 0; i < nx_; ++i) p[i] = nest_lb_[i] + (nest_ub_[i] - nest_lb_[i]) * sobol_.next();
    return p;
  }

  std::vector<Evaluation> evaluate_batch(const std::vector<std::vector<double>>& X) {
    const std::vector<std::vector<double>> G = evaluate_(X);
    if (G.size() != X.size())
      throw std::runtime_error("evaluator returned " + std::to_string(G.size()) + " results for " +
                               std::to_string(X.size()) + " points");
    std::vector<Evaluation> out(X.size());
    for (std::size_t i = 0; i < X.size(); ++i) {
      stats_.evaluations += ns_;
      if (G[i].empty()) {
        ++stats_.failures;
        out[i].ok = false;
        continue;
      }
      if (G[i].size() % ns_ != 0)
        throw std::runtime_error("evaluator returned " + std::to_string(G[i].size()) +
                                 " constraint values, not a multiple of the " + std::to_string(ns_) + " scenarios");
      const Criterion c = feasibility_criterion(G[i].data(), ns_, G[i].size() / ns_, weights_, opt_.threshold);
      out[i].ok = true;
      out[i].value = (opt_.criterion == Options::CVAR) ? c.cvar : c.var;
      out[i].prob = c.prob;
    }
    return out;
  }

  static Point make_point(const std::vector<double>& x, const Evaluation& ev) {
    Point p;
    p.x = x;
    p.value = ev.value;
    p.prob = ev.prob;
    return p;
  }

  void insert_live(const std::vector<double>& x, const Evaluation& ev) {
    if (ev.value <= 0.0) ++stats_.feasible;
    live_.insert(std::make_pair(ev.value, make_point(x, ev)));
  }

  // The live point with the largest criterion (last inserted among ties).
  double worst() const { return live_.rbegin()->first; }

  void kill_worst() {
    LiveSet::iterator last = live_.end();
    --last;
    dead_.push_back(last->second);
    live_.erase(last);
  }

  void display_row(std::size_t iteration) const {
    FormatGuard guard(os_);
    os_ << std::setw(7) << iteration
        << std::scientific << std::setprecision(4) << std::setw(15) << worst()
        << std::setw(6) << stats_.feasible << std::setw(9) << dead_.size()
        << std::scientific << std::setprecision(4) << std::setw(15) << factor_ << std::endl;
  }

  const std::vector<double>& lb_;
  const std::vector<double>& ub_;
  const std::vector<double>& weights_;
  const Options& opt_;
  const Evaluator& evaluate_;
  std::ostream& os_;
  std::size_t nx_, ns_;

  Sobol sobol_;
  Rng rng_;
  double factor_;
  std::vector<double> nest_lb_, nest_ub_, centre_, shape_;
  std::vector<double> z_, y_;

  LiveSet live_;
  std::vector<Point> dead_, discarded_;
  std::vector<IterationRecord> history_;
  Stats stats_;
  std::size_t iterations_;
  Clock::time_point t0_;
};

}  // namespace

NestedSampler::NestedSampler(std::vector<double> lb, std::vector<double> ub, Options options,
                             std::vector<double> scenario_weights)
  : lb_(std::move(lb)), ub_(std::move(ub)), weights_(std::move(scenario_weights)), options_(options) {
  if (lb_.empty() || lb_.size() != ub_.size())
    throw std::invalid_argument("lb and ub must have the same non-zero length");
  for (std::size_t i = 0; i < lb_.size(); ++i)
    if (!(lb_[i] < ub_[i])) throw std::invalid_argument("lb must be strictly below ub in every dimension");
  if (options_.num_live == 0) throw std::invalid_argument("num_live must be positive");
  if (options_.num_prop == 0) throw std::invalid_argument("num_prop must be positive");
  if (options_.criterion == Options::CVAR && !(options_.threshold > 0.0))
    throw std::invalid_argument("threshold must be positive for the CVaR criterion");

  for (std::size_t s = 0; s < weights_.size(); ++s)
    if (!(weights_[s] > 0.0)) throw std::invalid_argument("scenario weights must be positive");
}

Result NestedSampler::run(const Evaluator& evaluate, std::ostream& os) const {
  Run run(lb_, ub_, weights_, options_, evaluate, os);
  return run.execute();
}

}  // namespace ns
