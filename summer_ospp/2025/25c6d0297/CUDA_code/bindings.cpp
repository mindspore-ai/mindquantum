#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of std::vector <-> list
#include "HUBO.h"       // Include our class declarations

namespace py = pybind11;

// PYBIND11_MODULE's first argument (hubo_solver) must match the target name
// in CMakeLists.txt. This will be the name you `import` in Python.
PYBIND11_MODULE(hubo_solver, m) {
    m.doc() = "Python bindings for the HUBO CUDA solver (BSB and DSB)";

    // Bind the BSB class
    py::class_<BSB>(m, "BSB")
        // Bind the constructor
        .def(py::init<const std::vector<std::vector<int>>&, const std::vector<double>&, int, int, const std::string&>(),
             "Constructor for BSB solver",
             py::arg("J"), py::arg("coeff"), py::arg("iter"), py::arg("b_size"), py::arg("be"))
        
        // Bind public methods
        .def("initial", &BSB::initial, "Initialize spin and y values")
        .def("update", &BSB::update, "Run the BSB update iterations")
        .def("get_solution", &BSB::get_solution, "Get the optimal spin configuration (list of doubles in [-1, 1])")
        .def("get_energy", &BSB::get_energy, "Get the energy of the optimal solution");

    // Bind the DSB class
    py::class_<DSB>(m, "DSB")
        // Bind the constructor
        .def(py::init<const std::vector<std::vector<int>>&, const std::vector<double>&, int, int, const std::string&>(),
             "Constructor for DSB solver",
             py::arg("J"), py::arg("coeff"), py::arg("iter"), py::arg("b_size"), py::arg("be"))

        // Bind public methods
        .def("initial", &DSB::initial, "Initialize spin and y values")
        .def("update", &DSB::update, "Run the DSB update iterations")
        .def("get_solution", &DSB::get_solution, "Get the optimal spin configuration (list of doubles in [-1, 1])")
        .def("get_energy", &DSB::get_energy, "Get the energy of the optimal solution");
}