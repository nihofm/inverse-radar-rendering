#include <torch/extension.h>
#include "context.h"

// -------------------------------------------------------
// python bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Optix context class
    py::class_<OptixContext>(m, "OptixContext")
    .def(py::init<>())
    .def("build_bvh", py::overload_cast<torch::Tensor>(&OptixContext::build_bvh))
    .def("build_bvh", py::overload_cast<torch::Tensor, torch::Tensor>(&OptixContext::build_bvh))
    .def("raytrace", &OptixContext::raytrace)
    .def("visibility", &OptixContext::visibility)
    .def("hitgen_cosine_power", &OptixContext::hitgen_cosine_power)
    .def_readonly("AABB", &OptixContext::AABB);
}
