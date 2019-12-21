pub extern crate dlopen;

#[macro_use]
mod macros;

pub mod raw;

mod lib {
    use crate::raw::RawOpenCL;
    use dlopen::utils::platform_file_name;
    use dlopen::wrapper::Container;
    use std::env::var_os;
    use std::fmt::{self, Debug, Formatter};
    use std::sync::Arc;

    /// Safe interface for dynamically-loaded OpenCL bindings.
    ///
    /// This struct is just a wrapper around an `Arc` to an internal set of raw
    /// bindings, and it may be cloned relatively cheaply.
    #[derive(Clone)]
    pub struct OpenCL {
        lib: Arc<Container<RawOpenCL>>,
    }

    impl Debug for OpenCL {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            self.lib.fmt(f)
        }
    }

    impl OpenCL {
        /// Attempt to load the system OpenCL library.
        ///
        /// If the `OPENCL_LIBRARY` environment variable is set, it will be used
        /// to load the library (it may be set to either a path, or a library
        /// name). Otherwise, the library will be loaded from the system library
        /// path.
        ///
        /// If the library could not be loaded for any reason (e.g. not found,
        /// missing required symbols, etc) then this function will return an
        /// error. If the library can be opened, this call will succeed as long
        /// as all OpenCL 1.0 functions are present. Functions added in OpenCL
        /// 1.1 and later revisions will be loaded if present, but `dynamic_ocl`
        /// will attempt to gracefully downgrade if they're not available. This
        /// may result in some functionality being unavailable at runtime.
        ///
        /// # Example
        /// ```
        /// # use dynamic_ocl::OpenCL;
        /// match OpenCL::load() {
        ///     Ok(_) => println!("Successfully loaded OpenCL library!"),
        ///     Err(e) => eprintln!("Error loading OpenCL library: {:?}", e)   
        /// }
        /// ```
        pub fn load() -> Result<Self, dlopen::Error> {
            let name = var_os("OPENCL_LIBRARY").unwrap_or_else(|| platform_file_name("OpenCL"));
            let lib = Arc::new(unsafe { Container::load(name) }?);
            Ok(Self { lib })
        }

        /// Borrow the underlying raw bindings.
        pub fn raw(&self) -> &Container<RawOpenCL> {
            &self.lib
        }

        /// Consume self and return the underlying raw bindings.
        pub fn into_raw(self) -> Arc<Container<RawOpenCL>> {
            self.lib
        }
    }
}

pub mod device;
mod error;
pub mod platform;
pub mod queue;
pub mod util;

pub use error::*;
pub use lib::*;
