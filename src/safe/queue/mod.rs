mod types;

use crate::buffer::flags::HostAccess;
use crate::buffer::{AsBuffer, MemSafe};
use crate::device::Device;
use crate::kernel::{Kernel, KernelArgList};
use crate::raw::*;
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::ffi::c_void;
use std::fmt;
use std::fmt::{Debug, Formatter};
pub use types::*;

/// An OpenCL command queue
#[derive(PartialEq, Eq, Hash)]
pub struct Queue(pub(crate) cl_command_queue);

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = wrap_result!("clReleaseCommandQueue" => clReleaseCommandQueue(self.0)) {
                log::warn!("Error releasing OpenCL command queue {:?}: {:?}", self, e);
            }
        }
    }
}

impl Debug for Queue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl OclInfoInternal for Queue {
    type Param = cl_command_queue_info;
    const DEBUG_CONTEXT: &'static str = "clGetCommandQueueInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetCommandQueueInfo(
            self.0,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Queue {
    pub fn try_clone(&self) -> Result<Self> {
        unsafe {
            wrap_result!("clRetainCommandQueue" => clRetainCommandQueue(self.0))?;
            Ok(Self(self.0))
        }
    }

    /// Get the raw handle for this queue. Note that this handle is only a raw
    /// pointer and does not use RAII to ensure validity, so you must manually
    /// make sure that it's not released while still in use.
    pub fn raw(&self) -> cl_command_queue {
        self.0
    }

    /// Wrap the given raw command queue handle
    ///
    /// # Safety
    ///
    /// If the given handle is not a valid OpenCL queue, behavior is undefined.
    /// Additionally, the reference count must stay above zero until the wrapper
    /// is dropped (which will implicitly release the handle and decrement the
    /// reference count).
    pub unsafe fn from_raw(handle: cl_command_queue) -> Self {
        Self(handle)
    }

    info_funcs! {
        pub fn context_raw(&self) -> cl_context = CL_QUEUE_CONTEXT;
        pub fn device_raw(&self) -> cl_device_id = CL_QUEUE_DEVICE;
        pub fn reference_count(&self) -> cl_uint = CL_QUEUE_REFERENCE_COUNT;
        pub fn properties(&self) -> QueueProperties = CL_QUEUE_PROPERTIES;
        pub fn size(&self) -> cl_uint = CL_QUEUE_SIZE;
        pub fn device_default_raw(&self) -> cl_command_queue = CL_QUEUE_DEVICE_DEFAULT;
    }

    pub fn device(&self) -> Result<Device> {
        self.device_raw().map(Device)
    }

    pub fn device_default(&self) -> Result<Queue> {
        self.device_default_raw().map(Queue)
    }

    /// Begin a new buffer command
    pub fn buffer_cmd<'q, 'a, H: HostAccess, T: MemSafe>(
        &'q mut self,
        buffer: &'q mut dyn AsBuffer<'a, H, T>,
    ) -> BufferCmd<'q, 'a, H, T> {
        BufferCmd {
            queue: self,
            buffer,
            offset: None,
        }
    }

    /// Begin a new kernel execution command
    pub fn kernel_cmd<'q, T: KernelArgList, W: WorkDims>(
        &'q mut self,
        kernel: &'q mut Kernel<T>,
    ) -> KernelCmd<'q, T, W> {
        KernelCmd {
            queue: self,
            kernel,
            global_work_offset: None,
            local_work_size: None,
        }
    }
}
