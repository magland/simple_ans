#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "simple_ans/cpp/simple_ans.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_simple_ans, m)
{
    m.doc() = "Simple ANS (Asymmetric Numeral Systems) implementation";

    py::class_<simple_ans::EncodedData>(m, "EncodedData")
        .def(py::init<>())
        .def_readwrite("state", &simple_ans::EncodedData::state)
        .def_readwrite("bitstream", &simple_ans::EncodedData::bitstream)
        .def_readwrite("num_bits", &simple_ans::EncodedData::num_bits);

    m.def(
        "encode",
        [](py::array_t<int32_t> signal,
           py::array_t<uint32_t> symbol_counts,
           py::array_t<int32_t> symbol_values)
        {
            py::buffer_info signal_buf = signal.request();
            py::buffer_info counts_buf = symbol_counts.request();
            py::buffer_info values_buf = symbol_values.request();

            if (counts_buf.ndim != 1 || values_buf.ndim != 1)
            {
                throw std::runtime_error("symbol_counts and symbol_values must be 1-dimensional");
            }
            if (counts_buf.shape[0] != values_buf.shape[0])
            {
                throw std::runtime_error(
                    "symbol_counts and symbol_values must have the same length");
            }

            return simple_ans::encode(static_cast<const int32_t*>(signal_buf.ptr),
                                      signal_buf.size,
                                      static_cast<const uint32_t*>(counts_buf.ptr),
                                      static_cast<const int32_t*>(values_buf.ptr),
                                      counts_buf.shape[0]);
        },
        "Encode signal using ANS",
        py::arg("signal").noconvert(),
        py::arg("symbol_counts").noconvert(),
        py::arg("symbol_values").noconvert());

    m.def(
        "decode",
        [](uint32_t state,
           const std::vector<uint64_t>& bitstream,
           size_t num_bits,
           py::array_t<uint32_t> symbol_counts,
           py::array_t<int32_t> symbol_values,
           size_t n)
        {
            py::buffer_info counts_buf = symbol_counts.request();
            py::buffer_info values_buf = symbol_values.request();

            if (counts_buf.ndim != 1 || values_buf.ndim != 1)
            {
                throw std::runtime_error("symbol_counts and symbol_values must be 1-dimensional");
            }
            if (counts_buf.shape[0] != values_buf.shape[0])
            {
                throw std::runtime_error(
                    "symbol_counts and symbol_values must have the same length");
            }

            auto result = py::array_t<int32_t>(n);
            py::buffer_info result_buf = result.request();

            simple_ans::decode(static_cast<int32_t*>(result_buf.ptr),
                               n,
                               state,
                               bitstream.data(),
                               num_bits,
                               static_cast<const uint32_t*>(counts_buf.ptr),
                               static_cast<const int32_t*>(values_buf.ptr),
                               counts_buf.shape[0]);

            return result;
        },
        "Decode ANS-encoded signal",
        py::arg("state"),
        py::arg("bitstream"),
        py::arg("num_bits"),
        py::arg("symbol_counts").noconvert(),
        py::arg("symbol_values").noconvert(),
        py::arg("n"));

    m.def(
        "choose_symbol_counts",
        [](py::array_t<double> proportions, uint32_t L)
        {
            py::buffer_info props_buf = proportions.request();
            if (props_buf.ndim != 1)
            {
                throw std::runtime_error("proportions must be 1-dimensional");
            }

            auto result = py::array_t<uint32_t>(props_buf.shape[0]);
            py::buffer_info result_buf = result.request();

            simple_ans::choose_symbol_counts(static_cast<uint32_t*>(result_buf.ptr),
                                             static_cast<const double*>(props_buf.ptr),
                                             props_buf.shape[0],
                                             L);

            return result;
        },
        "Convert real-valued proportions into integer counts summing to L",
        py::arg("proportions").noconvert(),
        py::arg("L"));

    m.def(
        "add_one_test",
        [](py::array_t<int32_t> input)
        {
            py::buffer_info buf = input.request();
            auto result = py::array_t<int32_t>(buf.shape[0]);
            py::buffer_info result_buf = result.request();

            simple_ans::add_one_test(static_cast<int32_t*>(result_buf.ptr),
                                     static_cast<const int32_t*>(buf.ptr),
                                     buf.shape[0]);

            return result;
        },
        "Test function that adds 1 to each element of an array - for benchmarking data transfer",
        py::arg("input").noconvert());
}
