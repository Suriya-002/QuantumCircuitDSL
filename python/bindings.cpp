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

  m.def("simulate", &simulate<double>, py::arg("circuit"),
        "Simulate a circuit from the all-zero state.");
}
