#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "simple_ans/cpp/simple_ans.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_simple_ans, m) {
    m.doc() = "Simple ANS (Asymmetric Numeral Systems) implementation";

    py::class_<simple_ans::EncodedData>(m, "EncodedData")
        .def(py::init<>())
        .def_readwrite("state", &simple_ans::EncodedData::state)
        .def_readwrite("bitstream", &simple_ans::EncodedData::bitstream)
        .def_readwrite("num_bits", &simple_ans::EncodedData::num_bits);

    m.def("encode", &simple_ans::encode, "Encode signal using ANS",
          py::arg("signal"), py::arg("symbol_counts"), py::arg("symbol_values"));

    m.def("decode", &simple_ans::decode, "Decode ANS-encoded signal",
          py::arg("state"), py::arg("bitstream"), py::arg("num_bits"),
          py::arg("symbol_counts"), py::arg("symbol_values"), py::arg("n"));

    m.def("choose_symbol_counts", &simple_ans::choose_symbol_counts,
          "Convert real-valued proportions into integer counts summing to L",
          py::arg("proportions"), py::arg("L"));

    m.def("add_one_test", &simple_ans::add_one_test,
          "Test function that adds 1 to each element of an array - for benchmarking data transfer",
          py::arg("input"));
}
