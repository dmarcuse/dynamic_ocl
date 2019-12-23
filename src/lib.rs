//! Dynamic bindings to OpenCL
//!
//! This library provides dynamically-loaded bindings to the system OpenCL
//! library, allowing you to add optional OpenCL support to an application
//! without requiring it as a linker dependency. Two APIs are available - raw,
//! unsafe, one-to-one bindings to the OpenCL C API, and a higher-level, safe(r)
//! and more convenient Rust API.
//!
//! The safe Rust API is only provided with the `safe` feature flag set, which
//! is enabled by default. If you prefer only the unsafe raw bindings, you can
//! disable the feature flag by setting `default_features = false` for the
//! dependency in your project manifest.

pub extern crate dlopen;

#[macro_use]
mod macros;

pub mod raw;

#[cfg(feature = "safe")]
mod safe {
    pub mod context;
    pub mod device;
    mod error;
    pub mod platform;
    pub mod queue;
    pub mod util;
    pub use error::*;
}

#[cfg(feature = "safe")]
pub use safe::*;

use crate::raw::OpenCLVersion;

/// Attempt to load the system OpenCL library, if not already loaded.
///
/// This function will search for the OpenCL library using the absolute path or
/// library name specified by the `OPENCL_LIBRARY` environment variable if set,
/// and from the system library path using a platform-specific version of the
/// name `OpenCL` if not set.
///
/// Once the library has been found and opened, symbols will be loaded from it
/// and bound to the appropriate function pointers in the `raw` module. This
/// call will attempt to determine the OpenCL version supported by the library
/// based on which symbols are present or missing, which will be returned. Any
/// functions for which symbols could not be found will be replaced by a
/// shim that simply panics.
///
/// Calling any raw function before OpenCL has been loaded with this function
/// will implicitly call this function to load OpenCL, panicking on failure.
///
/// The result of this call (including implicit calls from raw function shims)
/// will be stored in static memory, and future calls will simply return the
/// stored result rather than attempting to load the library again. Since raw
/// function pointers are replaced by this call, there is no overhead for OpenCL
/// library calls once the functions have been bound.
pub fn load_opencl() -> std::result::Result<OpenCLVersion, &'static dlopen::Error> {
    raw::functions::load_opencl()
}
