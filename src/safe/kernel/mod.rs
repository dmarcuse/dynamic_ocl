mod types;

use crate::program::Program;
use crate::raw::*;
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::ffi::{c_void, CStr, CString};
use std::fmt::{self, Debug, Formatter};
pub use types::*;

#[derive(PartialEq, Eq, Hash)]
pub struct Kernel<T: KernelArgList> {
    handle: cl_kernel,
    args: T::Bound,
}

impl<T: KernelArgList> Drop for Kernel<T> {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = wrap_result!("clReleaseKernel" => clReleaseKernel(self.handle)) {
                log::warn!("Error releasing OpenCL kernel: {:?}: {:?}", self, e);
            }
        }
    }
}

impl<T: KernelArgList> Debug for Kernel<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl<T: KernelArgList> OclInfoInternal for Kernel<T> {
    type Param = cl_kernel_info;
    const DEBUG_CONTEXT: &'static str = "clGetKernelInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetKernelInfo(
            self.handle,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl<T: KernelArgList> Kernel<T> {
    /// Get a reference to the argument list of this kernel, allowing the values
    /// to be updated after kernel creation
    pub fn arguments(&mut self) -> &T::Bound {
        &self.args
    }

    info_funcs! {
        pub fn function_name(&self) -> CString = CL_KERNEL_FUNCTION_NAME;
        pub fn num_args(&self) -> cl_uint = CL_KERNEL_NUM_ARGS;
        pub fn reference_count(&self) -> cl_uint = CL_KERNEL_REFERENCE_COUNT;
        pub fn context_raw(&self) -> cl_context = CL_KERNEL_CONTEXT;
        pub fn program_raw(&self) -> cl_program = CL_KERNEL_PROGRAM;
        pub fn attributes(&self) -> CString = CL_KERNEL_ATTRIBUTES;
    }
}

impl Program {
    /// Create a kernel with a given name and set of arguments.
    pub fn create_kernel<T: KernelArgList>(&self, name: &CStr, args: T) -> Result<Kernel<T>> {
        unsafe {
            let mut err = CL_SUCCESS;
            let kernel = clCreateKernel(self.raw(), name.as_ptr(), &mut err as _);
            wrap_result!("clCreateKernel" => err)?;
            args.bind(kernel)
        }
    }
}
