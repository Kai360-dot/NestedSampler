#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "nested_sampler/nested_sampler.hpp"

namespace py = pybind11;

namespace {

typedef py::array_t<double, py::array::c_style | py::array::forcecast> DoubleArray;

// (N, nx) coordinates, (N,) criterion values, (N,) feasibility probabilities.
py::dict points_to_dict(const std::vector<ns::Point>& points, std::size_t nx) {
  const std::size_t n = points.size();
  py::array_t<double> x({n, nx});
  py::array_t<double> value(n);
  py::array_t<double> prob(n);
  double* px = x.mutable_data();
  double* pv = value.mutable_data();
  double* pp = prob.mutable_data();
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < nx; ++j) px[i * nx + j] = points[i].x[j];
    pv[i] = points[i].value;
    pp[i] = points[i].prob;
  }
  py::dict d;
  d["x"] = x;
  d["value"] = value;
  d["prob"] = prob;
  return d;
}

py::dict history_to_dict(const std::vector<ns::IterationRecord>& h) {
  const std::size_t n = h.size();
  py::array_t<std::size_t> iteration(n), feasible(n), dead(n);
  py::array_t<double> contour(n), factor(n);
  py::array_t<bool> box(n);
  for (std::size_t i = 0; i < n; ++i) {
    iteration.mutable_data()[i] = h[i].iteration;
    feasible.mutable_data()[i] = h[i].feasible;
    dead.mutable_data()[i] = h[i].dead;
    contour.mutable_data()[i] = h[i].contour;
    factor.mutable_data()[i] = h[i].factor;
    box.mutable_data()[i] = h[i].box_sampling;
  }
  py::dict d;
  d["iteration"] = iteration;
  d["contour"] = contour;
  d["feasible"] = feasible;
  d["dead"] = dead;
  d["factor"] = factor;
  d["box_sampling"] = box;
  return d;
}

// Run the sampler with a Python evaluator: callable(X) with X of shape (m, nx)
// returning an array of shape (m, ns, ng) or (m, ng); a NaN anywhere in a
// point's values marks that evaluation as failed.
py::dict run(std::vector<double> lb, std::vector<double> ub, std::vector<double> weights,
             const ns::Options& options, py::object evaluate) {
  const ns::NestedSampler sampler(lb, ub, options, weights);
  const std::size_t nx = sampler.dimension();
  const std::size_t nscen = sampler.num_scenarios();

  ns::Evaluator evaluator = [&](const std::vector<std::vector<double>>& X) {
    const std::size_t m = X.size();
    py::array_t<double> arr({m, nx});
    double* p = arr.mutable_data();
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t j = 0; j < nx; ++j) p[i * nx + j] = X[i][j];

    DoubleArray G = evaluate(arr).cast<DoubleArray>();
    std::size_t ng = 0;
    if (G.ndim() == 3 && static_cast<std::size_t>(G.shape(0)) == m && static_cast<std::size_t>(G.shape(1)) == nscen)
      ng = static_cast<std::size_t>(G.shape(2));
    else if (G.ndim() == 2 && static_cast<std::size_t>(G.shape(0)) == m && nscen == 1)
      ng = static_cast<std::size_t>(G.shape(1));
    else
      throw std::invalid_argument("evaluator must return an array of shape (" + std::to_string(m) + ", " +
                                  std::to_string(nscen) + ", ng)");
    if (ng == 0) throw std::invalid_argument("evaluator returned no constraint values");

    const std::size_t stride = nscen * ng;
    const double* g = G.data();
    std::vector<std::vector<double>> out(m);
    for (std::size_t i = 0; i < m; ++i) {
      bool failed = false;
      for (std::size_t k = 0; k < stride && !failed; ++k) failed = std::isnan(g[i * stride + k]);
      if (!failed) out[i].assign(g + i * stride, g + (i + 1) * stride);
    }
    return out;
  };

  ns::Result r;
  {
    py::scoped_ostream_redirect redirect(std::cout, py::module_::import("sys").attr("stdout"));
    r = sampler.run(evaluator, std::cout);
  }

  py::dict stats;
  stats["iterations"] = r.stats.iterations;
  stats["evaluations"] = r.stats.evaluations;
  stats["failures"] = r.stats.failures;
  stats["feasible"] = r.stats.feasible;
  stats["walltime"] = r.stats.walltime;

  py::dict d;
  d["status"] = static_cast<int>(r.status);
  d["stats"] = stats;
  d["live"] = points_to_dict(r.live, nx);
  d["dead"] = points_to_dict(r.dead, nx);
  d["discarded"] = points_to_dict(r.discarded, nx);
  d["history"] = history_to_dict(r.history);
  return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Nested sampling for feasibility analysis (port of MAGNUS NSFEAS)";

  py::class_<ns::Options> options(m, "Options");
  options.def(py::init<>())
      .def_readwrite("criterion", &ns::Options::criterion)
      .def_readwrite("threshold", &ns::Options::threshold)
      .def_readwrite("num_live", &ns::Options::num_live)
      .def_readwrite("num_prop", &ns::Options::num_prop)
      .def_readwrite("ell_mag", &ns::Options::ell_mag)
      .def_readwrite("ell_red", &ns::Options::ell_red)
      .def_readwrite("max_iter", &ns::Options::max_iter)
      .def_readwrite("max_err", &ns::Options::max_err)
      .def_readwrite("max_time", &ns::Options::max_time)
      .def_readwrite("seed", &ns::Options::seed)
      .def_readwrite("rng_skip", &ns::Options::rng_skip)
      .def_readwrite("display", &ns::Options::display)
      .def_readwrite("display_iter", &ns::Options::display_iter);

  py::enum_<ns::Options::Criterion>(options, "Criterion")
      .value("VAR", ns::Options::VAR)
      .value("CVAR", ns::Options::CVAR)
      .export_values();

  py::enum_<ns::Status>(m, "Status")
      .value("NORMAL", ns::NORMAL)
      .value("INTERRUPT", ns::INTERRUPT)
      .export_values();

  m.def("run", &run, py::arg("lb"), py::arg("ub"), py::arg("weights"), py::arg("options"), py::arg("evaluate"),
        "Run nested sampling; see nested_sampler.NestedSampler for the friendly interface.");
}
