#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "qcdsl/qcdsl.hpp"

namespace py = pybind11;
using namespace qcdsl;

using SV = Statevector<double>;

PYBIND11_MODULE(_qcdsl, m) {
  m.doc() = "qcdsl: C++17 quantum circuit compiler (native module)";
  m.attr("__version__") = QCDSL_VERSION_INFO;

  py::enum_<GateKind>(m, "GateKind")
      .value("I", GateKind::I)
      .value("X", GateKind::X)
      .value("Y", GateKind::Y)
      .value("Z", GateKind::Z)
      .value("H", GateKind::H)
      .value("S", GateKind::S)
      .value("Sdg", GateKind::Sdg)
      .value("T", GateKind::T)
      .value("Tdg", GateKind::Tdg)
      .value("RX", GateKind::RX)
      .value("RY", GateKind::RY)
      .value("RZ", GateKind::RZ)
      .value("CX", GateKind::CX)
      .value("CZ", GateKind::CZ)
      .value("SWAP", GateKind::SWAP)
      .value("MEASURE", GateKind::MEASURE);

  m.def("arity", &arity, py::arg("kind"),
        "Number of qubits a gate kind acts on.");
  m.def("is_parametric", &is_parametric, py::arg("kind"),
        "True for gates carrying a rotation angle.");
  m.def(
      "gate_name", [](GateKind k) { return std::string(to_string(k)); },
      py::arg("kind"), "The OpenQASM name of a gate kind.");

  py::class_<Gate>(m, "Gate")
      .def(py::init<GateKind, std::vector<Qubit>, double>(), py::arg("kind"),
           py::arg("qubits"), py::arg("param") = 0.0)
      .def_readonly("kind", &Gate::kind)
      .def_readonly("qubits", &Gate::qubits)
      .def_readonly("param", &Gate::param)
      .def("width", &Gate::width)
      .def("__repr__", [](const Gate& g) {
        std::string r = std::string("<Gate ") + to_string(g.kind) + " on [";
        for (std::size_t i = 0; i < g.qubits.size(); ++i) {
          r += std::to_string(g.qubits[i]);
          if (i + 1 < g.qubits.size()) {
            r += ", ";
          }
        }
        return r + "]>";
      });

  py::class_<Circuit>(m, "Circuit")
      .def(py::init<std::size_t>(), py::arg("num_qubits"))
      .def("add", py::overload_cast<const Gate&>(&Circuit::add),
           py::arg("gate"), py::return_value_policy::reference_internal)
      .def(
          "add",
          [](Circuit& self, GateKind k, std::vector<Qubit> qs, double p)
              -> Circuit& { return self.add(Gate(k, std::move(qs), p)); },
          py::arg("kind"), py::arg("qubits"), py::arg("param") = 0.0,
          py::return_value_policy::reference_internal)
      .def_property_readonly("num_qubits", &Circuit::num_qubits)
      .def("size", &Circuit::size)
      .def("depth", &Circuit::depth)
      .def("gates", &Circuit::gates)
      .def("gate_count_on", &Circuit::gate_count_on, py::arg("qubit"))
      .def("__len__", &Circuit::size)
      .def("__repr__", [](const Circuit& c) {
        return "<Circuit qubits=" + std::to_string(c.num_qubits()) +
               " size=" + std::to_string(c.size()) +
               " depth=" + std::to_string(c.depth()) + ">";
      });

  py::class_<SV>(m, "Statevector")
      .def(py::init<std::size_t>(), py::arg("num_qubits"))
      .def_property_readonly("num_qubits", &SV::num_qubits)
      .def_property_readonly("dim", &SV::dim)
      .def("amplitudes", &SV::amplitudes,
           "All 2**n amplitudes, index-ordered with qubit 0 as the least "
           "significant bit (Qiskit's convention).")
      .def("amplitude", &SV::amplitude, py::arg("index"))
      .def("probabilities", &SV::probabilities)
      .def("norm", &SV::norm)
      .def("apply", &SV::apply, py::arg("gate"))
      .def("run", &SV::run, py::arg("circuit"))
      .def("__len__", &SV::dim)
      .def("__repr__", [](const SV& sv) {
        return "<Statevector qubits=" + std::to_string(sv.num_qubits()) +
               " dim=" + std::to_string(sv.dim()) + ">";
      });

  py::class_<Dag::Node>(m, "DagNode")
      .def_readonly("gate", &Dag::Node::gate)
      .def_readonly("preds", &Dag::Node::preds)
      .def_readonly("succs", &Dag::Node::succs);

  py::class_<Dag>(m, "Dag")
      .def(py::init<const Circuit&>(), py::arg("circuit"))
      .def_property_readonly("num_qubits", &Dag::num_qubits)
      .def("size", &Dag::size)
      .def("num_edges", &Dag::num_edges)
      .def("depth", &Dag::depth)
      .def("node", &Dag::node, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("nodes", &Dag::nodes, py::return_value_policy::reference_internal)
      .def("frontier", &Dag::frontier)
      .def("layers", &Dag::layers)
      .def("topological_order", &Dag::topological_order)
      .def("random_topological_order", &Dag::random_topological_order,
           py::arg("seed"))
      .def("is_valid_schedule", &Dag::is_valid_schedule, py::arg("order"))
      .def("to_circuit", py::overload_cast<>(&Dag::to_circuit, py::const_))
      .def("to_circuit",
           py::overload_cast<const std::vector<std::size_t>&>(&Dag::to_circuit,
                                                              py::const_),
           py::arg("order"))
      .def("__len__", &Dag::size)
      .def("__repr__", [](const Dag& d) {
        return "<Dag nodes=" + std::to_string(d.size()) +
               " edges=" + std::to_string(d.num_edges()) +
               " depth=" + std::to_string(d.depth()) + ">";
      });

  py::class_<PassStats>(m, "PassStats")
      .def_readonly("pass_name", &PassStats::pass)
      .def_readonly("gates_before", &PassStats::gates_before)
      .def_readonly("gates_after", &PassStats::gates_after)
      .def_readonly("depth_before", &PassStats::depth_before)
      .def_readonly("depth_after", &PassStats::depth_after)
      .def("changed", &PassStats::changed)
      .def("__repr__", [](const PassStats& s) {
        return "<PassStats " + s.pass +
               " gates=" + std::to_string(s.gates_before) + "->" +
               std::to_string(s.gates_after) +
               " depth=" + std::to_string(s.depth_before) + "->" +
               std::to_string(s.depth_after) + ">";
      });

  py::class_<Pass, std::shared_ptr<Pass>>(m, "Pass")
      .def("name", &Pass::name)
      .def("__call__", &Pass::operator(), py::arg("circuit"));

  py::class_<CancelInversePairs, Pass, std::shared_ptr<CancelInversePairs>>(
      m, "CancelInversePairs")
      .def(py::init<>());
  py::class_<MergeRotations, Pass, std::shared_ptr<MergeRotations>>(
      m, "MergeRotations")
      .def(py::init<>());
  py::class_<RemoveIdentities, Pass, std::shared_ptr<RemoveIdentities>>(
      m, "RemoveIdentities")
      .def(py::init<double>(), py::arg("eps") = 1e-12);
  py::class_<DecomposeToCx, Pass, std::shared_ptr<DecomposeToCx>>(
      m, "DecomposeToCx")
      .def(py::init<>());

  py::class_<PassManager>(m, "PassManager")
      .def(py::init<>())
      .def("add", &PassManager::add, py::arg("pass"),
           py::return_value_policy::reference_internal)
      .def("run", &PassManager::run, py::arg("circuit"))
      .def("run_to_fixed_point", &PassManager::run_to_fixed_point,
           py::arg("circuit"), py::arg("max_sweeps") = 16)
      .def("stats", &PassManager::stats)
      .def("sweeps", &PassManager::sweeps)
      .def("__len__", &PassManager::size);

  py::class_<CouplingMap>(m, "CouplingMap")
      .def(py::init<std::size_t, const std::vector<std::pair<Qubit, Qubit>>&>(),
           py::arg("num_qubits"), py::arg("edges"))
      .def_static("line", &CouplingMap::line, py::arg("n"))
      .def_static("ring", &CouplingMap::ring, py::arg("n"))
      .def_static("grid", &CouplingMap::grid, py::arg("rows"), py::arg("cols"))
      .def_static("all_to_all", &CouplingMap::all_to_all, py::arg("n"))
      .def_property_readonly("num_qubits", &CouplingMap::num_qubits)
      .def("num_edges", &CouplingMap::num_edges)
      .def("edges", &CouplingMap::edges)
      .def("neighbours", &CouplingMap::neighbours, py::arg("qubit"))
      .def("degree", &CouplingMap::degree, py::arg("qubit"))
      .def("are_connected", &CouplingMap::are_connected, py::arg("a"),
           py::arg("b"))
      .def("distance", &CouplingMap::distance, py::arg("a"), py::arg("b"))
      .def("is_connected", &CouplingMap::is_connected)
      .def("__repr__", [](const CouplingMap& c) {
        return "<CouplingMap qubits=" + std::to_string(c.num_qubits()) +
               " edges=" + std::to_string(c.num_edges()) + ">";
      });

  py::class_<SabreOptions>(m, "SabreOptions")
      .def(py::init<>())
      .def_readwrite("lookahead_weight", &SabreOptions::lookahead_weight)
      .def_readwrite("lookahead_size", &SabreOptions::lookahead_size)
      .def_readwrite("decay_step", &SabreOptions::decay_step)
      .def_readwrite("trials", &SabreOptions::trials)
      .def_readwrite("layout_trials", &SabreOptions::layout_trials)
      .def_readwrite("scoring_trials", &SabreOptions::scoring_trials)
      .def_readwrite("seed", &SabreOptions::seed);

  py::class_<RoutingResult>(m, "RoutingResult")
      .def_readonly("circuit", &RoutingResult::circuit)
      .def_readonly("initial_layout", &RoutingResult::initial_layout)
      .def_readonly("final_layout", &RoutingResult::final_layout)
      .def_readonly("swaps_added", &RoutingResult::swaps_added)
      .def("__repr__", [](const RoutingResult& r) {
        return "<RoutingResult swaps=" + std::to_string(r.swaps_added) +
               " gates=" + std::to_string(r.circuit.size()) + ">";
      });

  py::class_<SabreRouter>(m, "SabreRouter")
      .def(py::init<CouplingMap, SabreOptions>(), py::arg("device"),
           py::arg("options") = SabreOptions())
      .def("device", &SabreRouter::device,
           py::return_value_policy::reference_internal)
      .def("route",
           py::overload_cast<const Circuit&>(&SabreRouter::route, py::const_),
           py::arg("circuit"))
      .def("route",
           py::overload_cast<const Circuit&, const Layout&>(&SabreRouter::route,
                                                            py::const_),
           py::arg("circuit"), py::arg("initial_layout"))
      .def("find_layout", &SabreRouter::find_layout, py::arg("circuit"),
           py::arg("iterations") = 3)
      .def("compile", &SabreRouter::compile, py::arg("circuit"),
           py::arg("iterations") = 3)
      .def("respects_device", &SabreRouter::respects_device, py::arg("circuit"))
      .def("trivial_layout", &SabreRouter::trivial_layout,
           py::arg("num_logical"));

  m.def("permute_index", &permute_index, py::arg("index"),
        py::arg("final_layout"),
        "Where amplitude `index` of the unrouted state vector lands in the "
        "routed one.");

  py::register_exception<QasmError>(m, "QasmError", PyExc_ValueError);

  m.def("to_qasm3", &to_qasm3, py::arg("circuit"),
        "Emit a circuit as OpenQASM 3.0 source.");
  m.def("from_qasm3", &from_qasm3, py::arg("source"),
        "Parse OpenQASM 3.0 source into a circuit. Raises QasmError.");

  m.def("simulate", &simulate<double>, py::arg("circuit"),
        "Simulate a circuit from the all-zero state.");
}
