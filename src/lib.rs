pub extern crate dlopen;

#[macro_use]
mod macros;

pub mod context;
pub mod device;
mod error;
pub mod platform;
pub mod queue;
pub mod raw;
pub mod util;
use crate::raw::OpenCLVersion;
pub use error::*;

/// Attempt to load the system OpenCL library, if not already loaded.
pub fn load_opencl() -> std::result::Result<OpenCLVersion, &'static dlopen::Error> {
    raw::functions::load_opencl()
}
