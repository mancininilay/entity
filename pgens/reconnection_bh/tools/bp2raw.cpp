/**
 * bp2raw: variables of an ADIOS2 BP file as raw binary files.
 *
 *   bp2raw <file.bp> <output directory> [variable ...]
 *
 * For each variable (all of them if none is named) and each step of the file
 * writes `<directory>/<variable>.<k>.raw` (k = 0, 1, ... the steps of the
 * file; Entity writes one step per file) in the type of the variable, and one
 * line per file in `<directory>/index.txt`:
 *   <variable> <k> <type> <number of elements> <dim 1> <dim 2> ...
 * Local arrays (one block per rank) are concatenated in block order; their
 * dimensions are given as the total count.
 *
 * Build: c++ -std=c++17 -O2 bp2raw.cpp $(adios2-config --cxx-flags --cxx-libs) -o bp2raw
 */
#include <adios2.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  std::string sanitize(std::string s) {
    for (auto& c : s) {
      if (c == '/' or c == ' ') {
        c = '_';
      }
    }
    return s;
  }

  template <typename T>
  bool dump(adios2::IO&        io,
            adios2::Engine&    engine,
            const std::string& name,
            const std::string& type,
            const std::string& dir,
            std::size_t        k,
            std::ofstream&     index) {
    if (type != adios2::GetType<T>()) {
      return false;
    }
    auto var = io.InquireVariable<T>(name);
    if (not var) {
      throw std::runtime_error("cannot inquire " + name);
    }
    std::vector<T>           data;
    std::vector<std::size_t> dims;
    const auto               shape_id = var.ShapeID();
    if (shape_id == adios2::ShapeID::GlobalValue or
        shape_id == adios2::ShapeID::LocalValue) {
      data.resize(1);
      engine.Get(var, data.data(), adios2::Mode::Sync);
    } else if (shape_id == adios2::ShapeID::GlobalArray) {
      const auto shape = var.Shape();
      dims.assign(shape.begin(), shape.end());
      std::size_t n = 1;
      for (const auto d : dims) {
        n *= d;
      }
      data.resize(n);
      if (n > 0) {
        engine.Get(var, data.data(), adios2::Mode::Sync);
      }
    } else {
      // local arrays: every block, in order
      const auto blocks = engine.BlocksInfo(var, engine.CurrentStep());
      for (const auto& block : blocks) {
        std::size_t n = 1;
        for (const auto d : block.Count) {
          n *= d;
        }
        std::vector<T> part(n);
        if (n > 0) {
          var.SetBlockSelection(block.BlockID);
          engine.Get(var, part.data(), adios2::Mode::Sync);
        }
        data.insert(data.end(), part.begin(), part.end());
      }
      dims = { data.size() };
    }
    const std::string path = dir + "/" + sanitize(name) + "." +
                             std::to_string(k) + ".raw";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
      throw std::runtime_error("cannot write " + path);
    }
    if (not data.empty() and
        std::fwrite(data.data(), sizeof(T), data.size(), f) != data.size()) {
      std::fclose(f);
      throw std::runtime_error("short write " + path);
    }
    std::fclose(f);
    index << sanitize(name) << " " << k << " " << type << " " << data.size();
    for (const auto d : dims) {
      index << " " << d;
    }
    index << "\n";
    return true;
  }

} // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: bp2raw <file.bp> <output directory> [variable ...]\n";
    return 2;
  }
  try {
    const std::string     file = argv[1], dir = argv[2];
    std::set<std::string> wanted;
    for (int i = 3; i < argc; ++i) {
      wanted.insert(argv[i]);
    }
    adios2::ADIOS adios;
    auto          io     = adios.DeclareIO("bp2raw");
    auto          engine = io.Open(file, adios2::Mode::Read);
    std::ofstream index(dir + "/index.txt");
    if (not index) {
      throw std::runtime_error("cannot write " + dir + "/index.txt");
    }
    std::size_t           k = 0;
    std::set<std::string> found;
    while (engine.BeginStep() == adios2::StepStatus::OK) {
      for (const auto& [name, info] : io.AvailableVariables()) {
        if (not wanted.empty() and wanted.count(name) == 0) {
          continue;
        }
        const auto type = info.at("Type");
        const bool ok   = dump<float>(io, engine, name, type, dir, k, index) or
                        dump<double>(io, engine, name, type, dir, k, index) or
                        dump<int32_t>(io, engine, name, type, dir, k, index) or
                        dump<uint32_t>(io, engine, name, type, dir, k, index) or
                        dump<int64_t>(io, engine, name, type, dir, k, index) or
                        dump<uint64_t>(io, engine, name, type, dir, k, index) or
                        dump<int16_t>(io, engine, name, type, dir, k, index) or
                        dump<uint16_t>(io, engine, name, type, dir, k, index) or
                        dump<int8_t>(io, engine, name, type, dir, k, index) or
                        dump<uint8_t>(io, engine, name, type, dir, k, index);
        if (not ok) {
          throw std::runtime_error("unsupported type " + type + " of " + name);
        }
        found.insert(name);
      }
      engine.EndStep();
      ++k;
    }
    engine.Close();
    for (const auto& name : wanted) {
      if (found.count(name) == 0) {
        throw std::runtime_error("variable not in the file: " + name);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "bp2raw: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
