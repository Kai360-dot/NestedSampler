cmake_minimum_required(VERSION 3.15)
project(nested_sampler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Core library (no external dependencies).
add_library(nested_sampler STATIC src/nested_sampler.cpp)
target_include_directories(nested_sampler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nested_sampler PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Keep floating-point results independent of fused multiply-add availability.
  target_compile_options(nested_sampler PUBLIC -ffp-contract=off)
endif()

# Python extension.
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
