//! Contexts
//!
//! An OpenCL context represents a group of one or more devices from the same
//! platform, allowing the sharing of programs and buffers.

use crate::device::Device;
use crate::queue::{Queue, QueueBuilder};
use crate::raw::{
    clGetContextInfo, clReleaseContext, clRetainContext, cl_context, cl_context_info, cl_device_id,
    cl_uint,
};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::ffi::c_void;
use std::fmt::{self, Debug, Formatter};

/// An OpenCL context
#[derive(PartialEq, Eq, Hash)]
pub struct Context(pub(crate) cl_context);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = wrap_result!("clReleaseContext" => clReleaseContext(self.0)) {
                log::warn!("Error releasing OpenCL context {:?}: {:?}", self, e);
            }
        }
    }
}

impl Debug for Context {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl OclInfoInternal for Context {
    type Param = cl_context_info;
    const DEBUG_CONTEXT: &'static str = "clGetContextInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetContextInfo(
            self.0,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Context {
    /// Create a new command queue using this context and the given device, with
    /// no special properties set.
    pub fn create_queue(&self, device: Device) -> Result<Queue> {
        QueueBuilder::new(self, &device).build()
    }

    /// Attempt to clone this context, using `clRetainContext` to ensure the
    /// context is not released while a wrapper still exists.
    pub fn try_clone(&self) -> Result<Self> {
        unsafe {
            wrap_result!("clRetainContext" => clRetainContext(self.0))?;
            Ok(Self(self.0))
        }
    }

    /// Get the raw handle for this context. Note that this handle is only a raw
    /// pointer and does not use RAII to ensure validity, so you must manually
    /// make sure that it's not released while still in use.
    pub fn raw(&self) -> cl_context {
        self.0
    }

    /// Wrap the given raw context handle
    ///
    /// # Safety
    ///
    /// If the given handle is not a valid OpenCL context, behavior is
    /// undefined. Additionally, the reference count must stay above zero until
    /// the wrapper is dropped (which will implicitly release the handle and
    /// decrement the reference count).
    pub unsafe fn from_raw(handle: cl_context) -> Self {
        Self(handle)
    }

    info_funcs! {
        pub fn reference_count(&self) -> cl_uint = CL_CONTEXT_REFERENCE_COUNT;
        pub fn num_devices(&self) -> cl_uint = CL_CONTEXT_NUM_DEVICES;
        pub fn devices_raw(&self) -> Vec<cl_device_id> = CL_CONTEXT_DEVICES;
        // TODO: CL_CONTEXT_PROPERTIES
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        Ok(self.devices_raw()?.into_iter().map(Device).collect())
    }
}
