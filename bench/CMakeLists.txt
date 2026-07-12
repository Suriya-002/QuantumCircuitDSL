add_executable(qcdsl_bench_optimise optimise.cpp)
target_link_libraries(qcdsl_bench_optimise PRIVATE qcdsl::qcdsl)
target_compile_options(qcdsl_bench_optimise PRIVATE -O2)

add_executable(qcdsl_bench_simulate simulate.cpp)
target_link_libraries(qcdsl_bench_simulate PRIVATE qcdsl::qcdsl)
# -O2 only. Deliberately NOT -march=native: the point is to measure the hand-
# written AVX-512 kernel against a scalar baseline the compiler is free to
# auto-vectorise however it likes. Handing the baseline -mavx512f as well would
# be fairer still -- and the README says so.
target_compile_options(qcdsl_bench_simulate PRIVATE -O2)
