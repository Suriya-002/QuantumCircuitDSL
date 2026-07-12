include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  GIT_SHALLOW    TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(qcdsl_tests
  test_gate.cpp
  test_circuit.cpp
  test_statevector.cpp
  test_dag.cpp
  test_passes.cpp
  test_qasm3.cpp
)

target_link_libraries(qcdsl_tests PRIVATE qcdsl::qcdsl GTest::gtest_main)

if(QCDSL_COVERAGE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(qcdsl_tests PRIVATE --coverage -O0 -g)
  target_link_options(qcdsl_tests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(qcdsl_tests)
