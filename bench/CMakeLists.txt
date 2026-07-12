add_executable(qcdsl_bench_optimise optimise.cpp)
target_link_libraries(qcdsl_bench_optimise PRIVATE qcdsl::qcdsl)
target_compile_options(qcdsl_bench_optimise PRIVATE -O2)
