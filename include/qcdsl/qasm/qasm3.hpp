#pragma once

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qcdsl/circuit.hpp"
#include "qcdsl/gate.hpp"

namespace qcdsl {

/// Thrown when a QASM source does not parse, with the offending line number.
class QasmError : public std::runtime_error {
 public:
  QasmError(std::size_t line, const std::string& what)
      : std::runtime_error("qasm3:" + std::to_string(line) + ": " + what),
        line_(line) {}

  [[nodiscard]] std::size_t line() const noexcept { return line_; }

 private:
  std::size_t line_;
};

namespace qasm {

struct Token {
  enum class Kind : std::uint8_t { Ident, Number, Punct, String, End };

  Kind kind = Kind::End;
  std::string text;
  double number = 0.0;
  std::size_t line = 1;
};

namespace lex {

inline bool is_ident_start(char c) {
  return (std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_' ||
         c == '$';
}

inline bool is_ident_char(char c) {
  return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_' ||
         c == '$';
}

inline bool starts_number(const std::string& s, std::size_t i) {
  if (std::isdigit(static_cast<unsigned char>(s[i])) != 0) {
    return true;
  }
  return s[i] == '.' && i + 1 < s.size() &&
         (std::isdigit(static_cast<unsigned char>(s[i + 1])) != 0);
}

/// Advance past whitespace, // comments and /* */ comments. Returns false when
/// nothing was consumed, which is the caller's signal that a token starts here.
inline bool skip_trivia(const std::string& s, std::size_t& i,
                        std::size_t& line) {
  const std::size_t before = i;
  const std::size_t n = s.size();

  while (i < n) {
    const char c = s[i];
    if (c == '\n') {
      ++line;
      ++i;
    } else if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      ++i;
    } else if (c == '/' && i + 1 < n && s[i + 1] == '/') {
      while (i < n && s[i] != '\n') {
        ++i;
      }
    } else if (c == '/' && i + 1 < n && s[i + 1] == '*') {
      const std::size_t opened = line;
      i += 2;
      while (i + 1 < n && (s[i] != '*' || s[i + 1] != '/')) {
        if (s[i] == '\n') {
          ++line;
        }
        ++i;
      }
      if (i + 1 >= n) {
        throw QasmError(opened, "unterminated block comment");
      }
      i += 2;
    } else {
      break;
    }
  }
  return i != before;
}

inline Token lex_string(const std::string& s, std::size_t& i,
                        std::size_t line) {
  const std::size_t start = ++i;  // skip the opening quote
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\n') {
      throw QasmError(line, "unterminated string");
    }
    ++i;
  }
  if (i >= s.size()) {
    throw QasmError(line, "unterminated string");
  }
  Token t{Token::Kind::String, s.substr(start, i - start), 0.0, line};
  ++i;  // skip the closing quote
  return t;
}

inline Token lex_ident(const std::string& s, std::size_t& i, std::size_t line) {
  const std::size_t start = i;
  while (i < s.size() && is_ident_char(s[i])) {
    ++i;
  }
  return Token{Token::Kind::Ident, s.substr(start, i - start), 0.0, line};
}

inline Token lex_number(const std::string& s, std::size_t& i,
                        std::size_t line) {
  const char* begin = s.c_str() + i;
  char* end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) {
    throw QasmError(line, "malformed number");
  }
  const std::size_t len = static_cast<std::size_t>(end - begin);
  Token t{Token::Kind::Number, s.substr(i, len), v, line};
  i += len;
  return t;
}

}  // namespace lex

/// Split QASM source into tokens.
inline std::vector<Token> tokenize(const std::string& src) {
  std::vector<Token> out;
  std::size_t i = 0;
  std::size_t line = 1;

  while (i < src.size()) {
    if (lex::skip_trivia(src, i, line)) {
      continue;
    }
    if (i >= src.size()) {
      break;
    }

    const char c = src[i];
    if (c == '"') {
      out.push_back(lex::lex_string(src, i, line));
    } else if (lex::is_ident_start(c)) {
      out.push_back(lex::lex_ident(src, i, line));
    } else if (lex::starts_number(src, i)) {
      out.push_back(lex::lex_number(src, i, line));
    } else {
      out.push_back({Token::Kind::Punct, std::string(1, c), 0.0, line});
      ++i;
    }
  }

  out.push_back({Token::Kind::End, "", 0.0, line});
  return out;
}

/// Recursive-descent parser for OpenQASM 3, restricted to the subset this
/// library can represent: a single qubit register, the stdgates.inc gates we
/// support, and measurement.
class Parser {
 public:
  explicit Parser(const std::string& src) : toks_(tokenize(src)) {}

  Circuit parse() {
    parse_header();
    parse_declarations();
    if (!qubits_declared_) {
      throw QasmError(peek().line, "no qubit register declared");
    }

    Circuit qc(num_qubits_);
    while (peek().kind != Token::Kind::End) {
      parse_statement(qc);
    }
    return qc;
  }

 private:
  // ---- token helpers -----------------------------------------------------

  [[nodiscard]] const Token& peek(std::size_t ahead = 0) const {
    const std::size_t k = pos_ + ahead;
    return k < toks_.size() ? toks_[k] : toks_.back();
  }

  const Token& take() {
    const Token& t = peek();
    if (pos_ + 1 < toks_.size()) {
      ++pos_;
    }
    return t;
  }

  [[nodiscard]] bool at_punct(const std::string& p) const {
    return peek().kind == Token::Kind::Punct && peek().text == p;
  }

  [[nodiscard]] bool at_ident(const std::string& s) const {
    return peek().kind == Token::Kind::Ident && peek().text == s;
  }

  void expect_punct(const std::string& p) {
    if (!at_punct(p)) {
      throw QasmError(peek().line,
                      "expected '" + p + "' but found '" + peek().text + "'");
    }
    take();
  }

  // ---- grammar -----------------------------------------------------------

  void parse_header() {
    if (at_ident("OPENQASM")) {
      take();
      if (peek().kind != Token::Kind::Number) {
        throw QasmError(peek().line, "expected a version after OPENQASM");
      }
      const double v = take().number;
      if (v < 3.0) {
        throw QasmError(peek().line, "only OpenQASM 3 is supported, got " +
                                         std::to_string(v));
      }
      expect_punct(";");
    }
    while (at_ident("include")) {
      take();
      if (peek().kind != Token::Kind::String) {
        throw QasmError(peek().line, "expected a filename after include");
      }
      take();
      expect_punct(";");
    }
  }

  void parse_declarations() {
    for (;;) {
      if (at_ident("qubit") || at_ident("qreg")) {
        parse_qubit_decl();
      } else if (at_ident("bit") || at_ident("creg")) {
        parse_bit_decl();
      } else {
        return;
      }
    }
  }

  void parse_qubit_decl() {
    const std::size_t line = peek().line;
    take();  // qubit | qreg
    std::size_t size = 1;
    if (at_punct("[")) {
      take();
      size = read_index();
      expect_punct("]");
    }
    if (peek().kind != Token::Kind::Ident) {
      throw QasmError(line, "expected a name for the qubit register");
    }
    const std::string name = take().text;
    // 'qreg q[2];' is the OpenQASM 2 spelling; the size trails the name.
    if (at_punct("[")) {
      take();
      size = read_index();
      expect_punct("]");
    }
    expect_punct(";");

    if (qubits_declared_) {
      throw QasmError(line, "only one qubit register is supported");
    }
    if (size == 0) {
      throw QasmError(line, "qubit register must not be empty");
    }
    qreg_ = name;
    num_qubits_ = size;
    qubits_declared_ = true;
  }

  void parse_bit_decl() {
    take();  // bit | creg
    if (at_punct("[")) {
      take();
      read_index();
      expect_punct("]");
    }
    if (peek().kind != Token::Kind::Ident) {
      throw QasmError(peek().line, "expected a name for the bit register");
    }
    creg_ = take().text;
    if (at_punct("[")) {
      take();
      read_index();
      expect_punct("]");
    }
    expect_punct(";");
  }

  void parse_statement(Circuit& qc) {
    const Token& t = peek();
    if (t.kind != Token::Kind::Ident) {
      throw QasmError(t.line, "expected a statement, found '" + t.text + "'");
    }

    // measure q[i];  |  measure q[i] -> c[j];
    if (t.text == "measure") {
      take();
      const Qubit q = read_qubit();
      if (at_punct("-")) {  // '->'
        take();
        expect_punct(">");
        read_creg_index();
      }
      expect_punct(";");
      qc.add(GateKind::MEASURE, {q});
      return;
    }
    // c[j] = measure q[i];
    if (!creg_.empty() && t.text == creg_) {
      take();
      if (at_punct("[")) {
        take();
        read_index();
        expect_punct("]");
      }
      expect_punct("=");
      if (!at_ident("measure")) {
        throw QasmError(peek().line, "expected 'measure' after '='");
      }
      take();
      const Qubit q = read_qubit();
      expect_punct(";");
      qc.add(GateKind::MEASURE, {q});
      return;
    }
    // barrier q[0], q[1];  -- accepted and dropped: it constrains scheduling,
    // not semantics, and this IR has no place to record it yet.
    if (t.text == "barrier") {
      take();
      while (!at_punct(";") && peek().kind != Token::Kind::End) {
        take();
      }
      expect_punct(";");
      return;
    }

    parse_gate(qc);
  }

  void parse_gate(Circuit& qc) {
    const std::size_t line = peek().line;
    const std::string name = take().text;

    const GateKind kind = gate_kind(name, line);

    double param = 0.0;
    if (at_punct("(")) {
      take();
      param = parse_expr();
      expect_punct(")");
      if (!is_parametric(kind)) {
        throw QasmError(line, "gate '" + name + "' takes no parameter");
      }
    } else if (is_parametric(kind)) {
      throw QasmError(line, "gate '" + name + "' needs an angle");
    }

    std::vector<Qubit> qs;
    qs.push_back(read_qubit());
    while (at_punct(",")) {
      take();
      qs.push_back(read_qubit());
    }
    expect_punct(";");

    if (qs.size() != arity(kind)) {
      throw QasmError(line, "gate '" + name + "' takes " +
                                std::to_string(arity(kind)) +
                                " qubit(s), got " + std::to_string(qs.size()));
    }
    qc.add(Gate(kind, qs, param));
  }

  [[nodiscard]] static GateKind gate_kind(const std::string& name,
                                          std::size_t line) {
    static const std::unordered_map<std::string, GateKind> kTable = {
        {"id", GateKind::I},    {"i", GateKind::I},      {"x", GateKind::X},
        {"y", GateKind::Y},     {"z", GateKind::Z},      {"h", GateKind::H},
        {"s", GateKind::S},     {"sdg", GateKind::Sdg},  {"t", GateKind::T},
        {"tdg", GateKind::Tdg}, {"rx", GateKind::RX},    {"ry", GateKind::RY},
        {"rz", GateKind::RZ},   {"cx", GateKind::CX},    {"CX", GateKind::CX},
        {"cz", GateKind::CZ},   {"swap", GateKind::SWAP}};
    const auto it = kTable.find(name);
    if (it == kTable.end()) {
      throw QasmError(line, "unsupported gate '" + name + "'");
    }
    return it->second;
  }

  std::size_t read_index() {
    if (peek().kind != Token::Kind::Number) {
      throw QasmError(peek().line, "expected an integer index");
    }
    const Token& t = take();
    if (t.number < 0 ||
        t.number != static_cast<double>(static_cast<long long>(t.number))) {
      throw QasmError(t.line, "index must be a non-negative integer");
    }
    return static_cast<std::size_t>(t.number);
  }

  Qubit read_qubit() {
    const Token& t = peek();
    if (t.kind != Token::Kind::Ident) {
      throw QasmError(t.line, "expected a qubit, found '" + t.text + "'");
    }
    if (t.text != qreg_) {
      throw QasmError(t.line, "unknown qubit register '" + t.text + "'");
    }
    take();
    expect_punct("[");
    const std::size_t idx = read_index();
    expect_punct("]");
    if (idx >= num_qubits_) {
      throw QasmError(t.line, "qubit index " + std::to_string(idx) +
                                  " out of range for a " +
                                  std::to_string(num_qubits_) +
                                  "-qubit register");
    }
    return idx;
  }

  void read_creg_index() {
    if (peek().kind != Token::Kind::Ident) {
      throw QasmError(peek().line, "expected a classical bit");
    }
    take();
    if (at_punct("[")) {
      take();
      read_index();
      expect_punct("]");
    }
  }

  // ---- angle expressions -------------------------------------------------
  //
  // Qiskit emits 'rz(pi/2) q[0];' as readily as 'rz(1.5707963267948966)', so an
  // angle is an expression, not a literal.

  double parse_expr() {
    double v = parse_term();
    for (;;) {
      if (at_punct("+")) {
        take();
        v += parse_term();
      } else if (at_punct("-")) {
        take();
        v -= parse_term();
      } else {
        return v;
      }
    }
  }

  double parse_term() {
    double v = parse_unary();
    for (;;) {
      if (at_punct("*")) {
        take();
        v *= parse_unary();
      } else if (at_punct("/")) {
        take();
        const double d = parse_unary();
        if (d == 0.0) {
          throw QasmError(peek().line, "division by zero in an angle");
        }
        v /= d;
      } else {
        return v;
      }
    }
  }

  double parse_unary() {
    if (at_punct("-")) {
      take();
      return -parse_unary();
    }
    if (at_punct("+")) {
      take();
      return parse_unary();
    }
    return parse_primary();
  }

  double parse_primary() {
    if (at_punct("(")) {
      take();
      const double v = parse_expr();
      expect_punct(")");
      return v;
    }
    if (peek().kind == Token::Kind::Number) {
      return take().number;
    }
    if (peek().kind == Token::Kind::Ident) {
      const Token& t = take();
      if (t.text == "pi" || t.text == "PI") {
        return 3.14159265358979323846;
      }
      if (t.text == "tau" || t.text == "TAU") {
        return 6.28318530717958647692;
      }
      if (t.text == "euler" || t.text == "E") {
        return 2.71828182845904523536;
      }
      throw QasmError(t.line, "unknown constant '" + t.text + "' in an angle");
    }
    throw QasmError(peek().line,
                    "expected an angle, found '" + peek().text + "'");
  }

  std::vector<Token> toks_;
  std::size_t pos_ = 0;
  std::string qreg_ = "q";
  std::string creg_;
  std::size_t num_qubits_ = 0;
  bool qubits_declared_ = false;
};

/// %.17g: seventeen significant digits round-trips a double exactly, and %g
/// drops the trailing zeros so 0.5 stays "0.5" instead of
/// "0.50000000000000000".
inline std::string format_angle(double v) {
  std::array<char, 32> buf{};
  std::snprintf(buf.data(), buf.size(), "%.17g", v);
  return {buf.data()};
}

}  // namespace qasm

/// Emit a circuit as OpenQASM 3.0.
inline std::string to_qasm3(const Circuit& qc) {
  bool measured = false;
  for (const Gate& g : qc.gates()) {
    if (g.kind == GateKind::MEASURE) {
      measured = true;
      break;
    }
  }

  std::ostringstream os;
  os << "OPENQASM 3.0;\n";
  os << "include \"stdgates.inc\";\n";
  if (measured) {
    os << "bit[" << qc.num_qubits() << "] c;\n";
  }
  os << "qubit[" << qc.num_qubits() << "] q;\n";

  for (const Gate& g : qc.gates()) {
    if (g.kind == GateKind::MEASURE) {
      os << "c[" << g.qubits[0] << "] = measure q[" << g.qubits[0] << "];\n";
      continue;
    }
    os << to_string(g.kind);
    if (is_parametric(g.kind)) {
      os << "(" << qasm::format_angle(g.param) << ")";
    }
    os << " ";
    for (std::size_t i = 0; i < g.qubits.size(); ++i) {
      os << "q[" << g.qubits[i] << "]";
      if (i + 1 < g.qubits.size()) {
        os << ", ";
      }
    }
    os << ";\n";
  }
  return os.str();
}

/// Parse OpenQASM 3.0 into a circuit. Throws QasmError on malformed input.
inline Circuit from_qasm3(const std::string& src) {
  qasm::Parser parser(src);
  return parser.parse();
}

}  // namespace qcdsl
