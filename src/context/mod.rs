mod types;

use crate::device::Device;
use crate::raw::{clGetContextInfo, cl_context, cl_context_info, cl_device_id, cl_uint};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::fmt::{self, Debug, Formatter};
use std::os::raw::c_void;
pub use types::*;

/// An OpenCL context
#[derive(PartialEq, Eq, Hash)]
pub struct Context(pub(crate) cl_context);

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
    info_funcs! {
        pub fn reference_count(&self) -> cl_uint = CL_CONTEXT_REFERENCE_COUNT;
        pub fn num_devices(&self) -> cl_uint = CL_CONTEXT_NUM_DEVICES;
        pub fn device_ids(&self) -> Vec<cl_device_id> = CL_CONTEXT_DEVICES;
        // TODO: CL_CONTEXT_PROPERTIES
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        Ok(self.device_ids()?.into_iter().map(Device).collect())
    }
}
