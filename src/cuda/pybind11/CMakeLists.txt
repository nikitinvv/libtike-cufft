find_package(pybind11)

add_library(ptychofft MODULE ptychofft.cxx)

set_target_properties(ptychofft PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(ptychofft PRIVATE libptychofft)
target_link_libraries(ptychofft PRIVATE pybind11::module)

install(TARGETS ptychofft LIBRARY DESTINATION "libtike/cufft")
