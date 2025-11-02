cmake_minimum_required(VERSION 3.18)

project(NaNovelty LANGUAGES CXX CUDA)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Define configurations
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(SRC_DIR "${CMAKE_SOURCE_DIR}")

file(GLOB_RECURSE SOURCES
    "${SRC_DIR}/*.cu"
    "${SRC_DIR}/*.cpp"
)
list(FILTER SOURCES EXCLUDE REGEX ".*/build/.*")


file(GLOB_RECURSE HEADERS
    "${SRC_DIR}/*.h"
    "${SRC_DIR}/*.hpp"
)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/CudaReduction
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Set target properties for Debug configuration
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 61
    LINK_LIBRARIES CUDA::cudart_static
    CUDA_SEPARABLE_COMPILATION ON
)

# Debug configuration
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:WIN32;WIN64;_DEBUG;_CONSOLE>
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-Xcompiler=/Od,/Zi,/W3>
)

target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:/DEBUG /INCREMENTAL>
)

# Release configuration
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:WIN32;WIN64;NDEBUG;_CONSOLE>
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:-Xcompiler=/O2,/W1>
)


add_link_options(
    "$<$<CONFIG:Release>:/DEBUG>"
    "$<$<CONFIG:Release>:/OPT:REF>"
    "$<$<CONFIG:Release>:/OPT:ICF>"
)

# target_link_options(${PROJECT_NAME} PRIVATE
    # $<$<CONFIG:Release>:/DEBUG>
# )


#Set subsystem to console
set_target_properties(${PROJECT_NAME} PROPERTIES
   LINK_FLAGS "/SUBSYSTEM:CONSOLE"
)


set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})


message("SOURCES:")
foreach(source ${SOURCES})
    message("  ${source}")
endforeach()

message("HEADERS:")
foreach(header ${HEADERS})
    message("  ${header}")
endforeach()
