cmake_minimum_required (VERSION 2.6)
project (stenfw)

set (stenfw_VERSION_MAJOR 1)
set (stenfw_VERSION_MINOR 0)

INCLUDE(CheckIncludeFile)
INCLUDE(CheckLibraryExists)
INCLUDE(CheckCSourceRuns)

option(HAVE_SINGLE "Use 4-byte float/integer computations" ON)
option(HAVE_DOUBLE "Use 8-byte double/longlong computations" OFF)

if (HAVE_SINGLE AND HAVE_DOUBLE)
	message(FATAL_ERROR "The 4-byte or 8-byte computations mode must be selected")
endif (HAVE_SINGLE AND HAVE_DOUBLE)

if ((NOT HAVE_SINGLE) AND (NOT HAVE_DOUBLE))
	message(FATAL_ERROR "The 4-byte or 8-byte computations mode must be selected")
endif ((NOT HAVE_SINGLE) AND (NOT HAVE_DOUBLE))

if (HAVE_SINGLE)
	add_definitions(-DSINGLE)
endif (HAVE_SINGLE)

if (HAVE_DOUBLE)
	add_definitions(-DDOUBLE)
endif (HAVE_DOUBLE)

option(HAVE_MPI "Enable parallel computations with MPI" ON)
option(HAVE_CUDA "Enable computations on CUDA-enabled GPUs" ON)
option(HAVE_CUDA_MAPPED "Enable use of host-mapped memory on GPUs" ON)
option(HAVE_CUDA_PINNED "Enabled use of host-pinned memory on GPUs" OFF)
option(HAVE_VISUALIZE "Enable additional sync-ups for data dumping and visualization" ON)

if (HAVE_MPI)
	find_package(MPI REQUIRED)
	SET(CMAKE_C_COMPILER mpicc)
	SET(CMAKE_CXX_COMPILER mpicxx)
	add_definitions(-DMPI)
endif (HAVE_MPI)

if (HAVE_CUDA)
	find_package(CUDA REQUIRED)
	add_definitions(-DCUDA)
endif (HAVE_CUDA)

if (HAVE_CUDA_MAPPED AND HAVE_CUDA_PINNED)
	message(FATAL_ERROR "HAVE_CUDA_MAPPED and HAVE_CUDA_PINNED cannot be used together")
endif (HAVE_CUDA_MAPPED AND HAVE_CUDA_PINNED)

if (HAVE_CUDA_MAPPED)
	if (NOT HAVE_CUDA)
		message(WARNING "HAVE_CUDA_MAPPED only has effect together with HAVE_CUDA")
	else (NOT HAVE_CUDA)
		add_definitions(-DCUDA_MAPPED)
	endif (NOT HAVE_CUDA)
endif (HAVE_CUDA_MAPPED)

if (HAVE_CUDA_PINNED)
	if (NOT HAVE_CUDA)
		message(WARNING "HAVE_CUDA_PINNED only has effect together with HAVE_CUDA")
	else (NOT HAVE_CUDA)
		add_definitions(-DCUDA_PINNED)
	endif (NOT HAVE_CUDA)
endif (HAVE_CUDA_PINNED)

if (HAVE_VISUALIZE)
	# UCAR Vapor
	find_library(HAVE_VDF libvdf.so /opt/vapor/lib)
	if (NOT HAVE_VDF)
		message(FATAL_ERROR "Cannot find libvdf.so needed for visualization subsystem")
	endif (NOT HAVE_VDF)
	find_library(HAVE_NETCDF libnetcdf.so /opt/vapor/lib)
	if (NOT HAVE_NETCDF)
		message(FATAL_ERROR "Cannot find libnetcdf.so needed for visualization subsystem")
	endif (NOT HAVE_NETCDF)
	find_library(HAVE_EXPAT libexpat.so /opt/vapor/lib)
	if (NOT HAVE_EXPAT)
		message(FATAL_ERROR "Cannot find libexpat.so needed for visualization subsystem")
	endif (NOT HAVE_EXPAT)
	find_library(HAVE_COMMON libcommon.so /opt/vapor/lib)
	if (NOT HAVE_COMMON)
		message(FATAL_ERROR "Cannot find libcommon.so needed for visualization subsystem")
	endif (NOT HAVE_COMMON)
	find_library(HAVE_PROJ libproj.so /opt/vapor/lib)
	if (NOT HAVE_PROJ)
		message(FATAL_ERROR "Cannot find libproj.so needed for visualization subsystem")
	endif (NOT HAVE_PROJ)
	set(Vapor_LIBRARIES ${Vapor_LIBRARIES} data2vdf ${HAVE_VDF} ${HAVE_NETCDF} ${HAVE_EXPAT} ${HAVE_COMMON} ${HAVE_PROJ})
	add_definitions(-DVISUALIZE)
	# ImageMagick
	find_package(ImageMagick COMPONENTS MagickCore)
	include_directories("${ImageMagick_INCLUDE_DIRS}")
endif (HAVE_VISUALIZE)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/data2vdf")
include_directories("${PROJECT_SOURCE_DIR}/generators/include")
include_directories("${PROJECT_SOURCE_DIR}/grid/include")
include_directories("${PROJECT_SOURCE_DIR}/timing")
include_directories("${PROJECT_SOURCE_DIR}/stencils/isum13pt")
include_directories("${PROJECT_SOURCE_DIR}/stencils/wave13pt")
include_directories("${PROJECT_SOURCE_DIR}/stencils/wave13pt_patus")
include_directories("${PROJECT_SOURCE_DIR}/tests")

# Check AVX presence
set(CMAKE_REQUIRED_FLAGS "-mavx")
check_c_source_runs("
	#include <immintrin.h>
	#include <x86intrin.h>
	int main() { __m256 a = { 0, 1, 1, 1, 1, 1, 1, 1}; return a[0]; }" HAVE_AVX)
set(CMAKE_REQUIRED_FLAGS)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-std=c99 -fopenmp")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fopenmp")

# Infrastructure
if (HAVE_VISUALIZE)
	add_library(data2vdf
		data2vdf/raw2vdf.cpp data2vdf/vdfcreate.cpp)
endif (HAVE_VISUALIZE)
add_library(generators
	generators/src/genirand.c generators/src/gensine3d.c)
add_library(grid
	grid/src/allocate.c grid/src/balanced.c	grid/src/chkvalid.c
	grid/src/gather.c grid/src/grid.c grid/src/multiply.c
	grid/src/overlap.c grid/src/scatter.c grid/src/setedges.c
	grid/src/setlinks.c grid/src/setsizes.c grid/src/subcpy.c)
add_library(timing
	timing/timing.c)

# Stencils
if (HAVE_CUDA)
	cuda_add_library(isum13pt
		stencils/isum13pt/isum13pt_cpu.c stencils/isum13pt/isum13pt_gpu.cu)
	cuda_add_library(wave13pt
		stencils/wave13pt/wave13pt_cpu.c stencils/wave13pt/wave13pt_gpu.cu)
	if (HAVE_AVX)
		cuda_add_library(wave13pt_patus
			stencils/wave13pt_patus/wave13pt_avx.c stencils/wave13pt_patus/wave13pt_avxfma4.c)
		SET_TARGET_PROPERTIES(wave13pt_patus PROPERTIES COMPILE_FLAGS "-mavx")
	endif (HAVE_AVX)
else (HAVE_CUDA)
	add_library(isum13pt
		stencils/isum13pt/isum13pt_cpu.c stencils/isum13pt/isum13pt_gpu.cu)
	add_library(wave13pt
		stencils/wave13pt/wave13pt_cpu.c stencils/wave13pt/wave13pt_gpu.cu)
	if (HAVE_AVX)
		add_library(wave13pt_patus
			stencils/wave13pt_patus/wave13pt_avx.c stencils/wave13pt_patus/wave13pt_avxfma4.c)
		SET_TARGET_PROPERTIES(wave13pt_patus PROPERTIES COMPILE_FLAGS "-mavx")
	endif (HAVE_AVX)
endif (HAVE_CUDA)

# Tests
if (HAVE_CUDA)
	cuda_add_executable(test_isum13pt
		tests/isum13pt/isum13pt_save.c tests/isum13pt/isum13pt_test.c
		tests/test_init.c tests/test_load.c tests/test_parse.c)
	cuda_add_executable(test_wave13pt
		tests/wave13pt/wave13pt_save.c tests/wave13pt/wave13pt_test.c
		tests/test_init.c tests/test_load.c tests/test_parse.c)
	if (HAVE_AVX)
		cuda_add_executable(test_wave13pt_patus
			tests/wave13pt_patus/wave13pt_patus_save.c tests/wave13pt_patus/wave13pt_patus_test.c
			tests/test_init.c tests/test_load.c tests/test_parse.c)
	endif (HAVE_AVX)
else (HAVE_CUDA)
	add_executable(test_isum13pt
		tests/isum13pt/isum13pt_save.c tests/isum13pt/isum13pt_test.c
		tests/test_init.c tests/test_load.c tests/test_parse.c)
	add_executable(test_wave13pt
		tests/wave13pt/wave13pt_save.c tests/wave13pt/wave13pt_test.c
		tests/test_init.c tests/test_load.c tests/test_parse.c)
	if (HAVE_AVX)
		add_executable(test_wave13pt_patus
			tests/wave13pt_patus/wave13pt_patus_save.c tests/wave13pt_patus/wave13pt_patus_test.c
			tests/test_init.c tests/test_load.c tests/test_parse.c)
	endif (HAVE_AVX)
endif (HAVE_CUDA)

target_link_libraries(test_isum13pt m isum13pt generators grid timing)
target_link_libraries(test_wave13pt m wave13pt generators grid timing)
if (HAVE_AVX)
	target_link_libraries(test_wave13pt_patus m wave13pt_patus generators grid timing)
endif (HAVE_AVX)

if (HAVE_VISUALIZE)
	target_link_libraries(test_isum13pt data2vdf ${ImageMagick_LIBRARIES} ${Vapor_LIBRARIES})
	target_link_libraries(test_wave13pt data2vdf ${Vapor_LIBRARIES})
	if (HAVE_AVX)
		target_link_libraries(test_wave13pt_patus data2vdf ${Vapor_LIBRARIES})
	endif (HAVE_AVX)
endif (HAVE_VISUALIZE)

if (HAVE_CUDA)
	target_link_libraries(test_isum13pt ${CUDA_CUDART_LIBRARY})
	target_link_libraries(test_wave13pt ${CUDA_CUDART_LIBRARY})
	if (HAVE_AVX)
		target_link_libraries(test_wave13pt_patus ${CUDA_CUDART_LIBRARY})
	endif (HAVE_AVX)
endif (HAVE_CUDA)
