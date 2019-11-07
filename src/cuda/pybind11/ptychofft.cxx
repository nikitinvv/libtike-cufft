#include <pybind11/pybind11.h>
#include "ptychofft.cuh"

namespace py = pybind11;

PYBIND11_MODULE(ptychofft, m){

  py::class_<ptychofft>(m, "ptychofft")
    .def(py::init<int, int, int, int, int, int>(),
      py::arg("ptheta"),
      py::arg("nz"),
      py::arg("n"),
      py::arg("nscan"),
      py::arg("detector_shape"),
      py::arg("probe_shape")
    )
    .def_readonly("ptheta", &ptychofft::ptheta)
    .def_readonly("nz", &ptychofft::nz)
    .def_readonly("n", &ptychofft::n)
    .def_readonly("nscan", &ptychofft::nscan)
    .def_readonly("ndet", &ptychofft::ndet)
    .def_readonly("nprb", &ptychofft::nprb)
    .def("fwd", &ptychofft::fwd)
    .def("adj", &ptychofft::adj)
    .def("free", &ptychofft::free)
    ;
}
