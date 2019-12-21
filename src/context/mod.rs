mod types;

use crate::device::Device;
use crate::raw::{cl_context, cl_context_info, cl_device_id, cl_uint};
use crate::util::sealed::OclInfoInternal;
use crate::{OpenCL, Result};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::os::raw::c_void;
pub use types::*;

/// An OpenCL context
#[derive(Clone)]
pub struct Context {
    pub(crate) ocl: OpenCL,
    pub(crate) id: cl_context,
}

impl Debug for Context {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Context {}

impl Hash for Context {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id as usize)
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
        self.ocl.raw().CL10.clGetContextInfo(
            self.id,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Context {
    info_funcs! {
        pub fn reference_count(&self) -> cl_uint = self.get_info_uint(CL_CONTEXT_REFERENCE_COUNT);
        pub fn num_devices(&self) -> cl_uint = self.get_info_uint(CL_CONTEXT_NUM_DEVICES);
        pub fn device_ids(&self) -> Vec<cl_device_id> = self.get_info_raw(CL_CONTEXT_DEVICES);
        // TODO: CL_CONTEXT_PROPERTIES
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        Ok(self
            .device_ids()?
            .into_iter()
            .map(|id| Device {
                id,
                ocl: self.ocl.clone(),
            })
            .collect())
    }
}
