cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

#python_add_library(pymsym MODULE src/msym.c WITH_SOABI)

# Add your entire library as a shared library
add_library(pymsym SHARED
    src/msym.c
    src/basis_function.c
    src/character_table.c
    src/context.c
    src/debug.c
    src/elements.c
    src/equivalence_set.c
    src/geometry.c
    src/linalg.c
    src/msym_error.c
    src/permutation.c
    src/point_group.c
    src/rsh.c
    src/subspace.c
    src/symmetrize.c
    src/symmetry.c
    src/symop.c
)

# Include directories for your project
target_include_directories(pymsym PRIVATE ${Python_INCLUDE_DIRS})

# Link against Python libraries
target_link_libraries(pymsym PRIVATE ${Python_LIBRARIES})

# Enable export of all symbols which is necessary for dynamic libraries
set_target_properties(pymsym PROPERTIES C_VISIBILITY_PRESET hidden)

# If the export file is necessary, configure generation
include(GenerateExportHeader)
generate_export_header(pymsym 
    EXPORT_MACRO_NAME MSYM_EXPORT
    STATIC_DEFINE MSYM_EXPORTS_BUILT_AS_STATIC
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/msym_EXPORTS.h
)

# Ensure your C code can find generated headers
target_include_directories(pymsym PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Installation path setup for skbuild
if(SKBUILD)
    # Install library into the Python package 'pymsym' under 'libs' subdirectory
    install(TARGETS pymsym LIBRARY DESTINATION pymsym)
else()
    install(TARGETS pymsym LIBRARY DESTINATION lib)
endif()
