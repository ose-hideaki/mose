cmake_minimum_required(VERSION 3.18)
project(autodiff_vbm_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# The extension statically compiles the unmodified VBMicrolensing 5.5 C++ source
# distributed with the installed `VBMicrolensing` wheel.  The integration relies
# on VBM's internal contour data structures, so the version is hard-pinned.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import pathlib, VBMicrolensing; print(pathlib.Path(VBMicrolensing.__file__).parent / 'lib')"
    OUTPUT_VARIABLE VBM_SOURCE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "from jax.extend import ffi; print(ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

if(NOT EXISTS "${VBM_SOURCE_DIR}/VBMicrolensingLibrary.cpp")
    message(FATAL_ERROR "VBMicrolensing C++ sources were not found; install `VBMicrolensing==5.5`")
endif()
if(NOT EXISTS "${JAX_FFI_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "JAX FFI headers were not found")
endif()

pybind11_add_module(_autodiff_vbm_native
    vbm_ffi.cpp
    "${VBM_SOURCE_DIR}/VBMicrolensingLibrary.cpp")
target_include_directories(_autodiff_vbm_native PRIVATE
    "${VBM_SOURCE_DIR}"
    "${JAX_FFI_INCLUDE_DIR}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_autodiff_vbm_native PRIVATE -O3 -march=native)
else()
    target_compile_options(_autodiff_vbm_native PRIVATE -O3)
endif()

if(DEFINED AUTODIFF_VBM_OUTPUT_DIR)
    set_target_properties(_autodiff_vbm_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${AUTODIFF_VBM_OUTPUT_DIR}")
endif()
