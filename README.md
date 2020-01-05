# dynamic_ocl

Experimental Rust OpenCL bindings, focusing on simplicity and safety. Currently very WIP, requiring Rust 1.41+ (beta toolchain at time of writing).

## Features

- Minimum overhead
- Support OpenCL versions up to 2.2
- Provides both direct bindings to the OpenCL C API, and higher level, safer, easier-to-use bindings
  - High level bindings can be disabled with a feature flag to avoid bloat if desired 
- OpenCL library dynamically loaded at runtime, allowing compilation of a single binary with optional OpenCL support and easier cross-compilation.
- Types are designed with safety in mind, taking advantage of Rust's type system to prevent various types of runtime errors, including:
  - Kernel argument type mismatch
  - Illegal buffer access (e.g. attempting to read from a `HOST_NO_ACCESS` buffer)
  - Usage of unsafe types with buffers (e.g. types that have invalid bit patterns that could be created by an OpenCL kernel)