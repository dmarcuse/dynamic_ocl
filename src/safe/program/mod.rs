mod types;

use crate::raw::{
    clGetProgramInfo, clReleaseProgram, clRetainProgram, cl_context, cl_program, cl_program_info,
    cl_uint,
};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use libc::size_t;
use std::ffi::{c_void, CString};
pub use types::*;

/// An OpenCL program
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Program(pub(crate) cl_program);

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = wrap_result!("clReleaseProgram" => clReleaseProgram(self.0)) {
                log::warn!("Error releasing OpenCL program: {:?}: {:?}", self, e);
            }
        }
    }
}

impl OclInfoInternal for Program {
    type Param = cl_program_info;
    const DEBUG_CONTEXT: &'static str = "clGetProgramInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetProgramInfo(
            self.0,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Program {
    /// Attempt to clone this program, using `clRetainProgram` to ensure the
    /// program is not released while a wrapper still exists.
    pub fn try_clone(&self) -> Result<Self> {
        unsafe {
            wrap_result!("clRetainProgram" => clRetainProgram(self.0))?;
            Ok(Self(self.0))
        }
    }

    info_funcs! {
        pub fn reference_count(&self) -> cl_uint = CL_PROGRAM_REFERENCE_COUNT;
        pub fn context_raw(&self) -> cl_context = CL_PROGRAM_CONTEXT;
        pub fn num_devices(&self) -> cl_uint = CL_PROGRAM_NUM_DEVICES;
        pub fn source(&self) -> CString = CL_PROGRAM_SOURCE;
        pub fn il(&self) -> Vec<u8> = CL_PROGRAM_IL;
        pub fn binary_sizes(&self) -> Vec<size_t> = CL_PROGRAM_BINARY_SIZES;
        // TODO: CL_PROGRAM_BINARIES
        pub fn num_kernels(&self) -> size_t = CL_PROGRAM_NUM_KERNELS;
        pub fn kernel_names(&self) -> CString = CL_PROGRAM_KERNEL_NAMES;
        pub fn scope_global_ctors_present(&self) -> bool = CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT;
        pub fn scope_global_dtors_present(&self) -> bool = CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT;
    }
}
