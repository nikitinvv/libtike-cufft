# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(ptychofft.i PROPERTIES CPLUSPLUS ON)
swig_add_library(ptychofft LANGUAGE python SOURCES ptychofft.i)
target_link_libraries(ptychofft PRIVATE libptychofft)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ptychofft.py" DESTINATION "libtike/cufft")
install(TARGETS ptychofft LIBRARY DESTINATION "libtike/cufft")
