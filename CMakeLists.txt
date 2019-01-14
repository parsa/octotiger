# Copyright (c) 2015 Thomas Heller
# Copyright (c) 2015 Dominic Marcello
# Copyright (c) 2018-2019 Parsa Amini
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(octotiger CXX)

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0003 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
################################################################################
# options
################################################################################
#option(OCTOTIGER_WITH_SILO "Enable support for Silo output" ON)
option(OCTOTIGER_WITH_GRAV_PAR "Enable parallelism in gravitational solver" OFF)
option(OCTOTIGER_WITH_RADIATION "Enable radiation transport solver" OFF)
option(OCTOTIGER_WITH_CUDA "Enable CUDA fmm kernels" OFF)

################################################################################
# Find required packages
################################################################################
find_package(HPX REQUIRED NO_CMAKE_PACKAGE_REGISTRY)
find_package(Silo REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Vc REQUIRED)
find_package(Boost REQUIRED)

add_library(octotiger::hdf5 INTERFACE IMPORTED)
set_property(TARGET octotiger::hdf5
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
set_property(TARGET octotiger::hdf5
  PROPERTY INTERFACE_LINK_LIBRARIES ${HDF5_LIBRARIES})

if(OCTOTIGER_WITH_CUDA)
  enable_language(CUDA)

  string(APPEND HPX_CUDA_CLANG_FLAGS " -fno-fast-math -Xcuda-ptxas -v")
endif()

################################################################################
# Set up Octotiger target
################################################################################
# Octotiger executable
add_hpx_executable(
  octotiger
  SOURCES ""
  HEADERS ""
  DEPENDENCIES
    Silo::silo Vc::Vc Boost::boost
)

target_include_directories(octotiger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(src)

if(USE_AVX2)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX2)
  else()
    target_compile_options(octotiger PRIVATE -mavx2)
  endif()
elseif(USE_AVX)
  if(MSVC)
    target_compile_options(octotiger PRIVATE /arch:AVX)
  else()
    target_compile_options(octotiger PRIVATE -mavx)
  endif()
endif()

# Handle CUDA
if(OCTOTIGER_WITH_CUDA)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_CUDA)
  set_property(TARGET octotiger PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Handle other options
if(OCTOTIGER_WITH_GRAV_PAR)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_GRAV_PAR)
endif()

if(OCTOTIGER_WITH_RADIATION)
  target_compile_definitions(octotiger PRIVATE OCTOTIGER_HAVE_RADIATION)
endif()

if(MSVC)
  # Enable solution folders for MSVC
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  target_compile_definitions(octotiger PRIVATE
    _USE_MATH_DEFINES
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    __restrict__=__restrict)

  target_compile_options(octotiger PRIVATE
    /wd4018
    /wd4068 # unknown pragma
    /wd4146
    /wd4244
    /wd4267
    /wd4334
    /wd4521
    /wd4800
    /wd4996)

  target_compile_options(octotiger PRIVATE /EHsc)

  # Ensure source files are shown in groups
  get_target_property(octotiger_SOURCES octotiger SOURCES)
  source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${octotiger_SOURCES})
else()
  target_link_libraries(octotiger octotiger::hdf5 dl z quadmath)
  
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    target_compile_options(octotiger PRIVATE
      -Wno-ignored-attributes -Wno-attributes -Wno-deprecated-declarations
      -Wno-unused-result -ffast-math)
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
      #set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ipo -Wunused-result")
      #set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast -ipo")
      target_compile_options(octotiger -Wno-attributes -Wno-deprecated)
  endif()
endif()

################################################################################
# Set up silo_compare target
################################################################################
add_executable(silo_compare src/compare/compare.cpp)

target_link_libraries(silo_compare Silo::silo)
if(NOT MSVC)
  target_link_libraries(silo_compare octotiger::hdf5 pthread dl z)
else()
  target_compile_options(silo_compare PRIVATE /EHsc)
endif()
