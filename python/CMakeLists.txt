include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_qcdsl bindings.cpp)
target_link_libraries(_qcdsl PRIVATE qcdsl::qcdsl)
target_compile_definitions(_qcdsl PRIVATE QCDSL_VERSION_INFO="${PROJECT_VERSION}")

# Drop the native module straight into an importable package directory inside
# the build tree, so `PYTHONPATH=build/python pytest` just works and the source
# tree stays clean.
set(QCDSL_PY_PKG ${CMAKE_BINARY_DIR}/python/qcdsl)

set_target_properties(_qcdsl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${QCDSL_PY_PKG})

add_custom_command(TARGET _qcdsl POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/qcdsl/__init__.py
          ${QCDSL_PY_PKG}/__init__.py
  COMMENT "Staging qcdsl Python package")
