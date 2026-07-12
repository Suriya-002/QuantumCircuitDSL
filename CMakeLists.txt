cmake_minimum_required(VERSION 3.20)

project(qcdsl
  VERSION 0.1.0
  DESCRIPTION "A C++17 quantum circuit compiler: DAG IR, optimisation passes, SIMD state-vector backend"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(QCDSL_BUILD_TESTS   "Build the C++ test suite"    ON)
option(QCDSL_BUILD_PYTHON  "Build the pybind11 module"   ON)
option(QCDSL_BUILD_BENCH   "Build the benchmark harness" OFF)
option(QCDSL_COVERAGE      "Instrument for gcov"         OFF)
option(QCDSL_OPENMP        "Use OpenMP for the parallel gate kernel" ON)

# ---------------------------------------------------------------------------
# The library itself is header-only: an INTERFACE target that carries the
# include path and the language standard, nothing to compile.
# ---------------------------------------------------------------------------
add_library(qcdsl INTERFACE)
add_library(qcdsl::qcdsl ALIAS qcdsl)

target_include_directories(qcdsl INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(qcdsl INTERFACE cxx_std_17)

# The AVX-512 kernel is compiled unconditionally via __attribute__((target(...)))
# and gated at RUN time by __builtin_cpu_supports. Deliberately NOT -mavx512f:
# that would let the compiler emit AVX-512 everywhere, and the binary would die
# with SIGILL on any machine without it -- including some CI runners.
if(QCDSL_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(qcdsl INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "qcdsl: OpenMP enabled (parallel gate kernel)")
  else()
    message(STATUS "qcdsl: OpenMP not found -- the parallel kernel falls back to serial")
  endif()
endif()

if(MSVC)
  target_compile_options(qcdsl INTERFACE /W4)
else()
  target_compile_options(qcdsl INTERFACE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(QCDSL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(QCDSL_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(QCDSL_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS qcdsl EXPORT qcdslTargets)
install(EXPORT qcdslTargets
  FILE qcdslTargets.cmake
  NAMESPACE qcdsl::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/qcdsl)
