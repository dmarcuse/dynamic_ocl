use super::{Kernel, UnboundKernel};
use crate::buffer::flags::HostAccess;
use crate::buffer::sealed::AsBufferInternal;
use crate::buffer::{AsBuffer, Buffer, MemSafe};
use crate::raw::*;
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use libc::size_t;
use std::ffi::c_void;
use std::ffi::CString;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomPinned;
use std::mem::{size_of, size_of_val};
use std::pin::Pin;
use tynm::type_name;

pub(crate) mod sealed {
    use super::{BindProject, KernelArgList};
    use crate::kernel::{Kernel, UnboundKernel};
    use crate::Result;
    use std::pin::Pin;

    pub trait KernelArgListInternal {
        fn bind(self, kernel: UnboundKernel, type_checks: bool) -> Result<Kernel<Self>>
        where
            Self: Sized + KernelArgList;
    }

    pub trait BindProjectInternal<'a> {
        fn project(self: Pin<&'a mut Self>) -> Self::Projected
        where
            Self: BindProject<'a>;
    }
}

pub trait KernelInfo: OclInfoInternal<Param = cl_kernel_info> + Sized {
    /// Get a reference to the unbound form of this kernel
    fn as_unbound(&self) -> &UnboundKernel;

    info_funcs! {
        fn function_name(&self) -> CString = CL_KERNEL_FUNCTION_NAME;
        fn num_args(&self) -> cl_uint = CL_KERNEL_NUM_ARGS;
        fn reference_count(&self) -> cl_uint = CL_KERNEL_REFERENCE_COUNT;
        fn context_raw(&self) -> cl_context = CL_KERNEL_CONTEXT;
        fn program_raw(&self) -> cl_program = CL_KERNEL_PROGRAM;
        fn attributes(&self) -> CString = CL_KERNEL_ATTRIBUTES;
    }

    fn arg_info(&self, idx: cl_uint) -> KernelArgInfo {
        let num_args = self.num_args().unwrap();
        assert!(
            idx < num_args,
            "index {} is out of range for kernel of arity {}",
            idx,
            num_args
        );

        KernelArgInfo {
            kernel: self.as_unbound(),
            idx,
        }
    }
}

#[derive(Clone, Copy)]
pub struct KernelArgInfo<'a> {
    kernel: &'a UnboundKernel,
    idx: cl_uint,
}

impl<'a> OclInfoInternal for KernelArgInfo<'a> {
    type Param = cl_kernel_arg_info;
    const DEBUG_CONTEXT: &'static str = "clGetKernelArgInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetKernelArgInfo(
            self.kernel.0,
            self.idx,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl<'a> Debug for KernelArgInfo<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

flag_enum! {
    pub enum ArgAddressQualifier(cl_kernel_arg_address_qualifier) {
        Global = CL_KERNEL_ARG_ADDRESS_GLOBAL,
        Local = CL_KERNEL_ARG_ADDRESS_LOCAL,
        Constant = CL_KERNEL_ARG_ADDRESS_CONSTANT,
        Private = CL_KERNEL_ARG_ADDRESS_PRIVATE
    }
}

flag_enum! {
    pub enum ArgAccessQualifier(cl_kernel_arg_access_qualifier) {
        ReadOnly = CL_KERNEL_ARG_ACCESS_READ_ONLY,
        WriteOnly = CL_KERNEL_ARG_ACCESS_WRITE_ONLY,
        ReadWrite = CL_KERNEL_ARG_ACCESS_READ_WRITE,
        None = CL_KERNEL_ARG_ACCESS_NONE
    }
}

bitfield! {
    pub struct ArgTypeQualifier(cl_kernel_arg_type_qualifier) {
        pub const CONST = CL_KERNEL_ARG_TYPE_CONST;
        pub const RESTRICT = CL_KERNEL_ARG_TYPE_RESTRICT;
        pub const VOLATILE = CL_KERNEL_ARG_TYPE_VOLATILE;
        pub const NONE = CL_KERNEL_ARG_TYPE_NONE;
    }
}

impl<'a> KernelArgInfo<'a> {
    info_funcs! {
        pub fn address_qualifier(&self) -> ArgAddressQualifier = CL_KERNEL_ARG_ADDRESS_QUALIFIER;
        pub fn access_qualifier(&self) -> ArgAccessQualifier = CL_KERNEL_ARG_ACCESS_QUALIFIER;
        pub fn type_name(&self) -> CString = CL_KERNEL_ARG_TYPE_NAME;
        pub fn type_qualifier(&self) -> ArgTypeQualifier = CL_KERNEL_ARG_TYPE_QUALIFIER;
        pub fn arg_name(&self) -> CString = CL_KERNEL_ARG_NAME;
    }
}

/// A trait implemented by types that can be used as an individual kernel
/// argument
pub trait KernelArg {
    /// The type of value which is passed to the `clSetKernelArg` call
    type ArgType;

    /// Check whether a given OpenCL C type (e.g. `float` or `ulong`) is
    /// compatible with this kernel argument type.
    fn is_param_type_compatible(c_type: &str) -> bool;

    /// Get the data of this kernel argument, as a size and value to be passed
    /// to `clSetKernelArg`
    fn as_raw_kernel_arg(&self) -> (size_t, &Self::ArgType);
}

// values can be used as individual kernel args
impl<T: MemSafe> KernelArg for T {
    type ArgType = T;

    fn is_param_type_compatible(c_type: &str) -> bool {
        T::is_param_type_compatible(c_type)
    }

    fn as_raw_kernel_arg(&self) -> (size_t, &T) {
        (size_of_val(self), &self)
    }
}

// buffers can be used as individual kernel args
impl<H: HostAccess, T: MemSafe> KernelArg for Buffer<'_, H, T> {
    type ArgType = cl_mem;

    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type
            .rsplitn(2, '*')
            .nth(1)
            .map(T::is_param_type_compatible)
            .unwrap_or(false)
    }

    fn as_raw_kernel_arg(&self) -> (size_t, &cl_mem) {
        (size_of::<cl_mem>(), &self.handle)
    }
}

/// A "bound" kernel argument which already has a value set, but can be updated.
///
/// # Safety
///
/// This type should only be used behind an immutable reference, as returned by
/// `Kernel::arguments`. It keeps a raw handle to the associatd kernel, whose
/// lifetime is not recorded in the type signature, so moving/mutating this type
/// is never safe outside of the provided inherent methods.
pub struct Bound<A: KernelArg> {
    _pinned: PhantomPinned,
    kernel: cl_kernel,
    index: cl_uint,
    value: A,
}

impl<K: KernelArg> Bound<K> {
    pub(crate) fn bind(
        kernel: &UnboundKernel,
        index: cl_uint,
        value: K,
        type_check: bool,
    ) -> Result<Self> {
        unsafe {
            let (size, ptr) = value.as_raw_kernel_arg();

            if type_check && SYSTEM_OPENCL_VERSION >= OpenCLVersion::CL12 {
                let arg_info = kernel.arg_info(index);

                match arg_info.type_name() {
                    Ok(s) if K::is_param_type_compatible(&s.to_string_lossy()) => {}
                    Ok(s) => {
                        panic!(
                            "Kernel argument type mismatch - OpenCL type {:?} is not compatible with {} for argument #{} ({:?}) of kernel {:?}",
                            s,
                            type_name::<K>(),
                            index,
                            arg_info,
                            kernel
                        );
                    }
                    Err(e) => {
                        log::warn!(
                            "Could not check type of argument #{} ({:?}) of kernel {:?}: {:?}",
                            index,
                            arg_info,
                            kernel,
                            e
                        );
                    }
                }
            }

            wrap_result!("clSetKernelArg" => clSetKernelArg(
                kernel.0,
                index,
                size,
                ptr as *const _ as _
            ))?;

            Ok(Self {
                _pinned: PhantomPinned,
                kernel: kernel.0,
                index,
                value,
            })
        }
    }

    /// Get a reference to the current value of this argument
    pub fn get(&self) -> &K {
        &self.value
    }

    /// Replace the current value of this argument with a new value, returning
    /// the original value if successful.
    pub fn replace(self: Pin<&mut Self>, value: K) -> Result<K> {
        unsafe {
            let (size, ptr) = value.as_raw_kernel_arg();

            wrap_result!("clSetKernelArg" => clSetKernelArg(
                self.kernel,
                self.index,
                size,
                ptr as *const _ as _
            ))?;

            Ok(std::mem::replace(
                &mut self.get_unchecked_mut().value,
                value,
            ))
        }
    }

    /// Set this argument to a new value.
    pub fn set(self: Pin<&mut Self>, value: K) -> Result<()> {
        unsafe {
            let (size, ptr) = value.as_raw_kernel_arg();

            wrap_result!("clSetKernelArg" => clSetKernelArg(
                self.kernel,
                self.index,
                size,
                ptr as *const _ as _
            ))?;

            Ok(())
        }
    }
}

pub trait BindProject<'a>: sealed::BindProjectInternal<'a> {
    type Projected: 'a;
}

/// A trait implemented by types that can be used as a complete set of kernel
/// arguments
pub trait KernelArgList: sealed::KernelArgListInternal + Sized {
    /// The "bound" form of this argument list, allowing updating arguments
    /// after the initial creation of the kernel.
    type Bound;

    /// The number of arguments specified by this argument list.
    const NUM_ARGS: usize;
}

kernel_arg_list_tuples! {
    (),
    (A),
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
    (A, B, C, D, E, F, G),
    (A, B, C, D, E, F, G, H),
    (A, B, C, D, E, F, G, H, I),
    (A, B, C, D, E, F, G, H, I, J),
    (A, B, C, D, E, F, G, H, I, J, K),
    (A, B, C, D, E, F, G, H, I, J, K, L),
    (A, B, C, D, E, F, G, H, I, J, K, L, M),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y),
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z),
}

impl<'a, H: HostAccess, T: MemSafe> AsBufferInternal<'a, H, T>
    for Pin<&mut Bound<Buffer<'a, H, T>>>
{
    fn as_buffer(&mut self) -> &Buffer<'a, H, T> {
        self.get()
    }
}

impl<'a, H: HostAccess, T: MemSafe> AsBuffer<'a, H, T> for Pin<&mut Bound<Buffer<'a, H, T>>> {}
