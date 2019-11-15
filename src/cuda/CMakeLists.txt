# Must set CUDA host compiler before enabling CUDA else it will use whatever it
# finds first. Alternatively, set the environment variable CUDAHOSTCXX
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
# find_package(NumPy REQUIRED)
# include_directories(${NumPy_INCLUDE_DIRS})

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(ptychofft.i PROPERTIES CPLUSPLUS ON)
swig_add_library(ptychofft LANGUAGE python SOURCES ptychofft.i ptychofft.cu)
swig_link_libraries(ptychofft cudart cufft)
target_compile_features(ptychofft PUBLIC cxx_std_11)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ptychofft.py" DESTINATION "libtike/cufft")
install(TARGETS ptychofft LIBRARY DESTINATION "libtike/cufft")
