cmake_minimum_required (VERSION 3.8)

project ("CudaCMake")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.h")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu" "${CMAKE_SOURCE_DIR}/src/*.cuh")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable (${PROJECT_NAME} ${cpu_source_files} ${gpu_source_files})