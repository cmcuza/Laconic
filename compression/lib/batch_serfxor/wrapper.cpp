// batch_pyserf: batched pybind11 binding for SerfXOR (one call per series,
// byte-identical to the per-value binding).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "utils/array.h"
#include "compressor/serf_xor_compressor.h"
#include "decompressor/serf_xor_decompressor.h"

namespace py = pybind11;

namespace {

std::vector<std::string> compress_blocks(const double *data, ssize_t n, int window_size,
                                         double max_diff, long adjust, int block_size) {
  std::vector<std::string> packs;
  if (block_size <= 0) block_size = 1000;
  packs.reserve(static_cast<size_t>((n + block_size - 1) / block_size));

  SerfXORCompressor compressor(window_size, max_diff, adjust);
  for (ssize_t begin = 0; begin < n; begin += block_size) {
    const ssize_t end = std::min<ssize_t>(begin + block_size, n);
    for (ssize_t i = begin; i < end; ++i) compressor.AddValue(data[i]);
    compressor.Close();
    Array<uint8_t> &bytes = compressor.compressed_bytes();
    packs.emplace_back(reinterpret_cast<const char *>(bytes.begin()),
                       static_cast<size_t>(bytes.length()));
  }
  return packs;
}

}

PYBIND11_MODULE(batch_pyserf, m) {
  m.doc() = "Batched SerfXOR binding - throughput measurement only, not used by any experiment";

  m.def(
      "compress",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
         int window_size, double max_diff, long adjust, int block_size) {
        auto info = values.request();
        const double *data = static_cast<const double *>(info.ptr);
        const ssize_t n = info.size;
        std::vector<std::string> packs;
        {
          py::gil_scoped_release release;
          packs = compress_blocks(data, n, window_size, max_diff, adjust, block_size);
        }
        py::list out;
        for (const std::string &pack : packs) out.append(py::bytes(pack));
        return out;
      },
      py::arg("values"), py::arg("window_size"), py::arg("max_diff"), py::arg("adjust"),
      py::arg("block_size") = 1000,
      "Compress a whole float64 array; returns one owned bytes object per block.");

  m.def(
      "compressed_size",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> values,
         int window_size, double max_diff, long adjust, int block_size) {
        auto info = values.request();
        const double *data = static_cast<const double *>(info.ptr);
        const ssize_t n = info.size;
        size_t total = 0;
        {
          py::gil_scoped_release release;
          for (const std::string &pack :
               compress_blocks(data, n, window_size, max_diff, adjust, block_size)) {
            total += pack.size();
          }
        }
        return total;
      },
      py::arg("values"), py::arg("window_size"), py::arg("max_diff"), py::arg("adjust"),
      py::arg("block_size") = 1000,
      "Total compressed size in bytes, without materialising the payload in Python.");

  m.def(
      "decompress",
      [](const std::vector<std::string> &packs, long adjust) {
        std::vector<double> out;
        {
          py::gil_scoped_release release;
          SerfXORDecompressor decompressor(adjust);
          for (const std::string &pack : packs) {
            std::vector<uint8_t> raw(pack.begin(), pack.end());
            Array<uint8_t> block(raw);
            std::vector<double> values = decompressor.Decompress(block);
            out.insert(out.end(), values.begin(), values.end());
          }
        }
        return py::array_t<double>(static_cast<py::ssize_t>(out.size()), out.data());
      },
      py::arg("packs"), py::arg("adjust"),
      "Decompress the blocks produced by compress(), in order.");
}
