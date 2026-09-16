cmake_minimum_required(VERSION 3.15)
project(nested_sampler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NS_BUILD_PYTHON "Build the Python extension module" ON)
option(NS_BUILD_CHECKS "Build the C++ reference checks (needs Boost headers)" OFF)

# Core library (no external dependencies).
add_library(nested_sampler STATIC src/nested_sampler.cpp)
target_include_directories(nested_sampler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nested_sampler PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Keep floating-point results independent of fused multiply-add availability.
  target_compile_options(nested_sampler PUBLIC -ffp-contract=off)
endif()

# Python extension.
if(NS_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
  endif()

  pybind11_add_module(_core src/bindings.cpp)
  target_link_libraries(_core PRIVATE nested_sampler)

  if(SKBUILD)
    install(TARGETS _core DESTINATION nested_sampler)
  else()
    # Plain CMake build: drop the module next to the Python package sources.
    add_custom_command(TARGET _core POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_core>
              ${CMAKE_CURRENT_SOURCE_DIR}/python/nested_sampler/)
  endif()
endif()

# Bit-compatibility checks against Boost and the standard library.
if(NS_BUILD_CHECKS)
  enable_testing()
  find_package(Boost REQUIRED)
  add_executable(reference_rng_check tests/cpp/reference_rng_check.cpp)
  target_link_libraries(reference_rng_check PRIVATE nested_sampler Boost::boost)
  add_test(NAME reference_rng_check COMMAND reference_rng_check)

  add_executable(cpp_example examples/cpp_example.cpp)
  target_link_libraries(cpp_example PRIVATE nested_sampler)
  add_test(NAME cpp_example COMMAND cpp_example)
endif()
