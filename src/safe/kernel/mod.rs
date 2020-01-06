//! Kernels

mod types;

use crate::program::Program;
use crate::raw::*;
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::ffi::{c_void, CStr};
use std::fmt::{self, Debug, Formatter};
use std::pin::Pin;
pub use types::*;

/// An OpenCL kernel, with arguments not yet set
#[derive(PartialEq, Eq, Hash)]
pub struct UnboundKernel(cl_kernel);

unsafe impl Send for UnboundKernel {}

/// An executable OpenCL kernel, with arguments set.
#[derive(PartialEq, Eq, Hash)]
pub struct Kernel<T: KernelArgList> {
    kernel: UnboundKernel,
    args: T::Bound,
}

unsafe impl<T: KernelArgList> Send for Kernel<T> {}

impl Drop for UnboundKernel {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = wrap_result!("clReleaseKernel" => clReleaseKernel(self.0)) {
                log::warn!("Error releasing OpenCL kernel: {:?}: {:?}", self, e);
            }
        }
    }
}

impl Debug for UnboundKernel {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl<T: KernelArgList> Debug for Kernel<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl OclInfoInternal for UnboundKernel {
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
            self.0,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl<T: KernelArgList> OclInfoInternal for Kernel<T> {
    type Param = <UnboundKernel as OclInfoInternal>::Param;
    const DEBUG_CONTEXT: &'static str = UnboundKernel::DEBUG_CONTEXT;

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        self.kernel.raw_info_internal(
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl KernelInfo for UnboundKernel {
    fn as_unbound(&self) -> &UnboundKernel {
        &self
    }
}

impl<T: KernelArgList> KernelInfo for Kernel<T> {
    fn as_unbound(&self) -> &UnboundKernel {
        &self.kernel
    }
}

impl UnboundKernel {
    /// Bind arguments to this kernel, performing type checks and calling
    /// `clSetKernelArg` to set values.
    ///
    /// When OpenCL 1.2+ features are available, this call will attempt to check
    /// the type of each argument before each call to `clSetKernelArg`. If the
    /// types appear to be incompatible, as determined by
    /// `KernelArg::is_param_type_compatible`, it will panic. If this behavior
    /// isn't desired, `bind_arguments_unchecked` can be used instead.
    pub fn bind_arguments<T: KernelArgList>(self, arguments: T) -> Result<Kernel<T>> {
        let required = self.num_args()? as usize;
        let supplied = T::NUM_ARGS;
        assert_eq!(
            required, supplied,
            "kernel arity mismatch - kernel requires {} arguments, but {} were specified",
            required, supplied
        );
        arguments.bind(self, true)
    }

    /// Bind arguments to this kernel, without performing assertions for number
    /// or type of arguments.
    ///
    /// # Safety
    ///
    /// This function is not unsafe in terms of memory safety, but it should
    /// still be used with care, as it can lead to unexpected results if the
    /// argument types don't match those of the OpenCL code.
    pub fn bind_arguments_unchecked<T: KernelArgList>(self, arguments: T) -> Result<Kernel<T>> {
        arguments.bind(self, false)
    }

    pub fn raw(&self) -> cl_kernel {
        self.0
    }
}

impl<T: KernelArgList> Kernel<T> {
    /// Get mutable references to the arguments of this kernel.
    ///
    /// For safety reasons, the bound arguments are pinned, and must not be
    /// moved.
    pub fn arguments<'a>(&'a mut self) -> <T::Bound as BindProject<'a>>::Projected
    where
        T::Bound: BindProject<'a>,
    {
        unsafe {
            use sealed::BindProjectInternal;
            let bound: Pin<&'a mut T::Bound> = Pin::new_unchecked(&mut self.args);
            BindProjectInternal::project(bound)
        }
    }
}

impl Program {
    /// Create a kernel with a given name.
    pub fn create_kernel(&self, name: &CStr) -> Result<UnboundKernel> {
        unsafe {
            let mut err = CL_SUCCESS;
            let kernel = clCreateKernel(self.raw(), name.as_ptr(), &mut err as _);
            wrap_result!("clCreateKernel" => err)?;
            Ok(UnboundKernel(kernel))
        }
    }
}
