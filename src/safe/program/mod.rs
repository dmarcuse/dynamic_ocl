//! Programs
//!
//! In OpenCL, a `Program` exports kernels that can be executed to perform work
//! on specialized hardware.

mod types;

use crate::context::Context;
use crate::device::Device;
use crate::raw::{
    clGetProgramBuildInfo, clGetProgramInfo, clReleaseProgram, clRetainProgram, cl_context,
    cl_device_id, cl_program, cl_program_build_info, cl_program_info, cl_uint,
};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use libc::size_t;
use std::ffi::{c_void, CString};
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
pub use types::*;

/// An OpenCL program
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Program(pub(crate) cl_program);

unsafe impl Send for Program {}
unsafe impl Sync for Program {}

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

    /// Get the raw handle for this program. Note that this handle is only a raw
    /// pointer and does not use RAII to ensure validity, so you must manually
    /// make sure that it's not released while still in use.
    pub fn raw(&self) -> cl_program {
        self.0
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

    /// Get program build info for a given device
    pub fn build_info(&self, Device(device): Device) -> Result<ProgramBuildInfo> {
        let context = ManuallyDrop::new(Context(self.context_raw()?));

        assert!(
            context.devices_raw()?.contains(&device),
            "program context does not include given device"
        );

        Ok(ProgramBuildInfo {
            device,
            program: self.0,
            program_ref: PhantomData,
        })
    }
}

pub struct ProgramBuildInfo<'a> {
    device: cl_device_id,
    program: cl_program,
    program_ref: PhantomData<&'a Program>,
}

impl OclInfoInternal for ProgramBuildInfo<'_> {
    type Param = cl_program_build_info;
    const DEBUG_CONTEXT: &'static str = "clGetProgramBuildInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetProgramBuildInfo(
            self.program,
            self.device,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl ProgramBuildInfo<'_> {
    info_funcs! {
        pub fn status(&self) -> ProgramBuildStatus = CL_PROGRAM_BUILD_STATUS;
        pub fn options(&self) -> CString = CL_PROGRAM_BUILD_OPTIONS;
        pub fn log(&self) -> CString = CL_PROGRAM_BUILD_LOG;
        pub fn binary_type(&self) -> ProgramBinaryType = CL_PROGRAM_BINARY_TYPE;
        pub fn global_variable_total_size(&self) -> size_t = CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE;
    }
}

impl Debug for ProgramBuildInfo<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}
