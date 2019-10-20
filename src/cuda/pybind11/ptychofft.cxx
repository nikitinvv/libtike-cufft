#include <pybind11/pybind11.h>
#include "ptychofft.cuh"

namespace py = pybind11;

// Get the array pointer from a cupy array.
size_t get_gpu_ptr(py::handle cupy_array)
{
  py::dict info = cupy_array.attr("__cuda_array_interface__");
  py::tuple data = info["data"];
  return data[0].cast<size_t>();
  // printf("%zu", (size_t)address);
}

PYBIND11_MODULE(ptychofft, m){
  m.doc() = "A module for ptychography solvers.";

  py::class_<ptychofft>(m, "PtychoCuFFT")
    .def(py::init<int, int, int, int, int, int, int>()
    // py::arg("detector_shape"), py::arg("probe_shape"), py::arg("nscan"),
    // py::arg("nz"), py::arg("n"), py::arg("ntheta") = 1
    )

    .def("__enter__", [](ptychofft& self){
      return self;
    })

    .def("__exit__", [](ptychofft& self, py::args args){
      self.free();
    })

    .def("fwd", [](ptychofft& self, py::handle farfield, py::handle probe,
                   py::handle scan, py::handle psi){
      self.fwd(
        get_gpu_ptr(farfield),
        get_gpu_ptr(probe),
        get_gpu_ptr(scan),
        get_gpu_ptr(psi)
      );
    },
    py::arg("farplane"), py::arg("probe"), py::arg("scan"), py::arg("psi")
    )

    .def("adj", [](ptychofft& self, py::handle farfield, py::handle probe,
                   py::handle scan, py::handle psi){
      self.adj(
        get_gpu_ptr(farfield),
        get_gpu_ptr(probe),
        get_gpu_ptr(scan),
        get_gpu_ptr(psi),
        0
      );
    },
    py::arg("farplane"), py::arg("probe"), py::arg("scan"), py::arg("psi")
    )
    .def("adj_probe", [](ptychofft& self, py::handle farfield, py::handle probe,
                         py::handle scan, py::handle psi){
      self.adj(
        get_gpu_ptr(farfield),
        get_gpu_ptr(probe),
        get_gpu_ptr(scan),
        get_gpu_ptr(psi),
        1
      );
    },
    py::arg("farplane"), py::arg("probe"), py::arg("scan"), py::arg("psi")
    )
    ;
}
