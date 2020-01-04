use crate::buffer::flags::HostAccess;
use crate::buffer::{Buffer, MemSafe};
use crate::kernel::Kernel;
use crate::raw::{clSetKernelArg, cl_kernel, cl_mem, cl_uint};
use crate::safe::kernel::types::sealed::BindProjectInternal;
use crate::Result;
use libc::size_t;
use std::ffi::c_void;
use std::marker::PhantomPinned;
use std::mem::{size_of, size_of_val};
use std::pin::Pin;

pub(crate) mod sealed {
    use super::{BindProject, KernelArgList};
    use crate::kernel::Kernel;
    use crate::raw::cl_kernel;
    use crate::Result;
    use std::pin::Pin;

    pub trait KernelArgListInternal {
        fn bind(self, kernel: cl_kernel) -> Result<Kernel<Self>>
        where
            Self: Sized + KernelArgList;
    }

    pub trait BindProjectInternal<'a> {
        fn project(self: Pin<&'a mut Self>) -> Self::Projected
        where
            Self: BindProject<'a>;
    }
}

/// A trait implemented by types that can be used as an individual kernel
/// argument
pub trait KernelArg {
    /// The type of value which is passed to the `clSetKernelArg` call
    type ArgType;

    /// Get the data of this kernel argument, as a size and value to be passed
    /// to `clSetKernelArg`
    fn as_raw_kernel_arg(&self) -> (size_t, &Self::ArgType);
}

// values can be used as individual kernel args
impl<T: MemSafe> KernelArg for T {
    type ArgType = T;

    fn as_raw_kernel_arg(&self) -> (size_t, &T) {
        (size_of_val(self), &self)
    }
}

// buffers can be used as individual kernel args
impl<H: HostAccess, T: MemSafe> KernelArg for Buffer<'_, H, T> {
    type ArgType = cl_mem;

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
    pub(crate) fn bind(kernel: cl_kernel, index: cl_uint, value: K) -> Result<Self> {
        // TODO: check type of argument against OpenCL kernel parameter type
        unsafe {
            let (size, ptr) = value.as_raw_kernel_arg();

            wrap_result!("clSetKernelArg" => clSetKernelArg(
                kernel,
                index,
                size,
                ptr as *const _ as _
            ))?;

            Ok(Self {
                _pinned: PhantomPinned,
                kernel,
                index,
                value,
            })
        }
    }
}

pub trait BindProject<'a>: BindProjectInternal<'a> {
    type Projected: 'a;
}

/// A trait implemented by types that can be used as a complete set of kernel
/// arguments
pub trait KernelArgList: sealed::KernelArgListInternal + Sized {
    /// The "bound" form of this argument list, allowing updating arguments
    /// after the initial creation of the kernel.
    type Bound;
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
