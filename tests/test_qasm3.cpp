#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/qasm/qasm3.hpp"
#include "qcdsl/sim/statevector.hpp"

namespace {

using qcdsl::Circuit;
using qcdsl::Gate;
using qcdsl::GateKind;
using qcdsl::QasmError;
using qcdsl::Qubit;

constexpr double kTol = 1e-12;
const double kPi = 3.14159265358979323846;

Circuit random_circuit(std::size_t n, std::size_t m, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::vector<GateKind> one = {GateKind::I,   GateKind::X, GateKind::Y,
                                     GateKind::Z,   GateKind::H, GateKind::S,
                                     GateKind::Sdg, GateKind::T, GateKind::Tdg};
  const std::vector<GateKind> rot = {GateKind::RX, GateKind::RY, GateKind::RZ};
  const std::vector<GateKind> two = {GateKind::CX, GateKind::CZ,
                                     GateKind::SWAP};
  std::uniform_real_distribution<double> ang(-6.0, 6.0);

  Circuit qc(n);
  for (std::size_t i = 0; i < m; ++i) {
    const auto roll = rng() % 100;
    if (n >= 2 && roll < 30) {
      std::uniform_int_distribution<std::size_t> pq(0, n - 1);
      Qubit a = pq(rng);
      Qubit b = pq(rng);
      while (b == a) {
        b = pq(rng);
      }
      qc.add(two[rng() % two.size()], {a, b});
    } else if (roll < 60) {
      qc.add(rot[rng() % rot.size()], {rng() % n}, ang(rng));
    } else {
      qc.add(one[rng() % one.size()], {rng() % n});
    }
  }
  return qc;
}

void ExpectSameState(const Circuit& a, const Circuit& b) {
  const qcdsl::Statevector<> sa = qcdsl::simulate(a);
  const qcdsl::Statevector<> sb = qcdsl::simulate(b);
  ASSERT_EQ(sa.dim(), sb.dim());
  for (std::size_t i = 0; i < sa.dim(); ++i) {
    EXPECT_NEAR(std::abs(sa.amplitude(i) - sb.amplitude(i)), 0.0, kTol)
        << "amplitude " << i;
  }
}

// --------------------------------------------------------------------------
// Emit
// --------------------------------------------------------------------------

TEST(Qasm3, EmitsAHeaderAndARegister) {
  Circuit qc(2);
  qc.add(GateKind::H, {0}).add(GateKind::CX, {0, 1});
  const std::string src = qcdsl::to_qasm3(qc);
  EXPECT_NE(src.find("OPENQASM 3.0;"), std::string::npos);
  EXPECT_NE(src.find("include \"stdgates.inc\";"), std::string::npos);
  EXPECT_NE(src.find("qubit[2] q;"), std::string::npos);
  EXPECT_NE(src.find("h q[0];"), std::string::npos);
  EXPECT_NE(src.find("cx q[0], q[1];"), std::string::npos);
  // NB: "qubit[2] q;" contains "bit[" as a substring, so anchor on the newline.
  EXPECT_EQ(src.find("\nbit["), std::string::npos)
      << "no measurement, so no classical register";
}

TEST(Qasm3, DeclaresAClassicalRegisterOnlyWhenMeasuring) {
  Circuit qc(1);
  qc.add(GateKind::MEASURE, {0});
  const std::string src = qcdsl::to_qasm3(qc);
  EXPECT_NE(src.find("bit[1] c;"), std::string::npos);
  EXPECT_NE(src.find("c[0] = measure q[0];"), std::string::npos);
}

// %.17g, not fixed precision: 0.5 must stay "0.5", and an awkward double must
// keep every digit it needs to come back exactly.
TEST(Qasm3, WritesAnglesThatRoundTripExactly) {
  const std::vector<double> angles = {0.5,   -1.0 / 3.0, kPi,
                                      1e-17, -6.0,       2.6367562034437613};
  for (const double a : angles) {
    Circuit qc(1);
    qc.add(GateKind::RZ, {0}, a);
    const Circuit back = qcdsl::from_qasm3(qcdsl::to_qasm3(qc));
    ASSERT_EQ(back.size(), 1u);
    EXPECT_EQ(back.gates()[0].param, a) << "angle " << a << " did not survive";
  }
}

// --------------------------------------------------------------------------
// Parse
// --------------------------------------------------------------------------

TEST(Qasm3, ParsesAMinimalProgram) {
  const Circuit qc = qcdsl::from_qasm3(R"(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
  )");
  EXPECT_EQ(qc.num_qubits(), 2u);
  ASSERT_EQ(qc.size(), 2u);
  EXPECT_EQ(qc.gates()[0].kind, GateKind::H);
  EXPECT_EQ(qc.gates()[1].kind, GateKind::CX);
}

// Qiskit writes 'rz(pi/2) q[0];' as happily as it writes a float, so an angle
// is an expression, not a literal.
TEST(Qasm3, EvaluatesAngleExpressions) {
  const Circuit qc = qcdsl::from_qasm3(R"(
    OPENQASM 3.0;
    qubit[1] q;
    rz(pi/2) q[0];
    rx(-pi) q[0];
    ry(2*pi/4 + 1) q[0];
    rz(-(3 - 1)/2) q[0];
    rx(tau) q[0];
  )");
  ASSERT_EQ(qc.size(), 5u);
  EXPECT_NEAR(qc.gates()[0].param, kPi / 2, 1e-15);
  EXPECT_NEAR(qc.gates()[1].param, -kPi, 1e-15);
  EXPECT_NEAR(qc.gates()[2].param, kPi / 2 + 1, 1e-15);
  EXPECT_NEAR(qc.gates()[3].param, -1.0, 1e-15);
  EXPECT_NEAR(qc.gates()[4].param, 2 * kPi, 1e-15);
}

TEST(Qasm3, SkipsCommentsAndTolerantWhitespace) {
  const Circuit qc = qcdsl::from_qasm3(
      "OPENQASM 3.0; // line comment\n"
      "/* a\n block\n comment */\n"
      "qubit[1] q;   h    q[ 0 ] ;\n");
  EXPECT_EQ(qc.size(), 1u);
}

TEST(Qasm3, AcceptsMeasurementInBothSpellings) {
  const Circuit a = qcdsl::from_qasm3(
      "OPENQASM 3.0; qubit[1] q; bit[1] c; c[0] = measure q[0];");
  const Circuit b = qcdsl::from_qasm3(
      "OPENQASM 3.0; qubit[1] q; bit[1] c; measure q[0] -> c[0];");
  EXPECT_EQ(a.size(), 1u);
  EXPECT_EQ(b.size(), 1u);
  EXPECT_EQ(a.gates()[0].kind, GateKind::MEASURE);
  EXPECT_EQ(b.gates()[0].kind, GateKind::MEASURE);
}

// A barrier constrains scheduling, not semantics. Accept it, drop it, say so.
TEST(Qasm3, AcceptsAndDropsBarriers) {
  const Circuit qc = qcdsl::from_qasm3(
      "OPENQASM 3.0; qubit[2] q; h q[0]; barrier q[0], q[1]; h q[1];");
  EXPECT_EQ(qc.size(), 2u);
}

TEST(Qasm3, AcceptsTheOpenQasm2RegisterSpelling) {
  const Circuit qc = qcdsl::from_qasm3("qreg q[3]; creg c[3]; h q[2];");
  EXPECT_EQ(qc.num_qubits(), 3u);
  EXPECT_EQ(qc.size(), 1u);
}

// --------------------------------------------------------------------------
// Errors -- a parser that accepts nonsense is worse than no parser.
// --------------------------------------------------------------------------

TEST(Qasm3, RejectsMalformedPrograms) {
  const std::vector<std::string> bad = {
      "qubit[2] q; nonsense q[0];",       // unknown gate
      "qubit[2] q; h q[0], q[1];",        // wrong arity
      "qubit[2] q; cx q[0];",             // wrong arity
      "qubit[2] q; h q[5];",              // index out of range
      "qubit[2] q; h r[0];",              // unknown register
      "qubit[2] q; h q[0]",               // missing semicolon
      "h q[0];",                          // no register declared
      "qubit[0] q; h q[0];",              // empty register
      "qubit[2] q; qubit[2] r; h q[0];",  // two registers
      "qubit[2] q; rz q[0];",             // rotation without an angle
      "qubit[2] q; h(0.5) q[0];",         // parameter on a fixed gate
      "qubit[2] q; rz(1/0) q[0];",        // division by zero
      "qubit[2] q; rz(nope) q[0];",       // unknown constant
      "qubit[1] q; /* unterminated",      // bad comment
      "OPENQASM 2.0; qubit[1] q;",        // wrong version
      "qubit[1] q; h q[-1];",             // negative index
  };
  for (const std::string& src : bad) {
    EXPECT_THROW((void)qcdsl::from_qasm3(src), QasmError)
        << "accepted: " << src;
  }
}

TEST(Qasm3, ErrorsCarryALineNumber) {
  try {
    (void)qcdsl::from_qasm3("qubit[1] q;\nh q[0];\nbogus q[0];\n");
    FAIL() << "expected a QasmError";
  } catch (const QasmError& e) {
    EXPECT_EQ(e.line(), 3u);
    EXPECT_NE(std::string(e.what()).find("bogus"), std::string::npos);
  }
}

// --------------------------------------------------------------------------
// Round-trip
// --------------------------------------------------------------------------

TEST(Qasm3, RoundTripsEveryGateKind) {
  Circuit qc(2);
  for (const GateKind k :
       {GateKind::I, GateKind::X, GateKind::Y, GateKind::Z, GateKind::H,
        GateKind::S, GateKind::Sdg, GateKind::T, GateKind::Tdg}) {
    qc.add(k, {0});
  }
  qc.add(GateKind::RX, {1}, 0.31)
      .add(GateKind::RY, {1}, -1.7)
      .add(GateKind::RZ, {0}, 2.9)
      .add(GateKind::CX, {0, 1})
      .add(GateKind::CZ, {1, 0})
      .add(GateKind::SWAP, {0, 1});

  const Circuit back = qcdsl::from_qasm3(qcdsl::to_qasm3(qc));
  ASSERT_EQ(back.size(), qc.size());
  for (std::size_t i = 0; i < qc.size(); ++i) {
    EXPECT_EQ(back.gates()[i].kind, qc.gates()[i].kind) << "gate " << i;
    EXPECT_EQ(back.gates()[i].qubits, qc.gates()[i].qubits) << "gate " << i;
    EXPECT_EQ(back.gates()[i].param, qc.gates()[i].param) << "gate " << i;
  }
  ExpectSameState(qc, back);
}

TEST(Qasm3, RoundTripsRandomCircuitsExactly) {
  for (std::uint64_t seed = 0; seed < 40; ++seed) {
    const Circuit qc = random_circuit(1 + (seed % 5), 50, seed);
    const Circuit back = qcdsl::from_qasm3(qcdsl::to_qasm3(qc));
    ASSERT_EQ(back.size(), qc.size()) << "seed " << seed;
    EXPECT_EQ(back.depth(), qc.depth()) << "seed " << seed;
    ExpectSameState(qc, back);
  }
}

TEST(Qasm3, EmitIsStableUnderReparsing) {
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    const Circuit qc = random_circuit(4, 40, seed);
    const std::string once = qcdsl::to_qasm3(qc);
    const std::string twice = qcdsl::to_qasm3(qcdsl::from_qasm3(once));
    EXPECT_EQ(once, twice) << "seed " << seed;
  }
}

}  // namespace
