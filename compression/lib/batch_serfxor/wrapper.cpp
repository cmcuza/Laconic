// batch_pyserf - a batched pybind11 binding for SerfXOR.
//
// WHY THIS EXISTS
// ---------------
// The shipped `pyserf` binding exposes only `add_value(double)`, so compressing
// a series from Python costs one pybind11 call per value - measured at ~307 ns,
// about 31 ms per 100K values. That is binding overhead, not codec time
// (`SerfXORCompressor` itself does 100K values in well under a millisecond), and
// in a throughput comparison against SZ3/MixPiece/TerseTS - all of which take a
// whole buffer in one call - it would be reported as if SerfXOR were slow.
//
// This module hands the whole array across once and runs the identical per-block
// loop in C++. Same compressor, same window, same block boundaries, same
// `Close()` per block, so the bytes are identical; only the per-value Python
// round trip is gone. `tests`/the benchmark assert that byte equality rather
// than assuming it.
//
// It also fixes an aliasing trap by construction. Upstream `get()` is bound with
// `py::return_value_policy::reference`, handing back a reference to the
// compressor's own buffer: a pack held across further encoder use is no longer
// the pack that was produced, and both `to_bytes()` and `decompress()` hang on
// the stale handle. Here every block is copied into an owned `py::bytes` before
// the compressor is touched again, so packs can be collected safely.
//
// SCOPE: throughput measurement only. Nothing in the experiment pipeline imports
// it - `compression/lib/serfxor.py` still drives every result in `results/`.
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

// The block loop of compression/lib/serfxor.py, verbatim: ONE compressor for the
// whole series, Close() at every block boundary, the block's bytes taken before
// the next value is added.
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

}  // namespace

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
          // No Python object is touched inside, so the GIL can go - which also
          // means a caller can time this without interpreter interference.
          py::gil_scoped_release release;
          packs = compress_blocks(data, n, window_size, max_diff, adjust, block_size);
        }
        py::list out;
        for (const std::string &pack : packs) out.append(py::bytes(pack));
        return out;   // owned copies: safe to hold, unlike upstream get()
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
