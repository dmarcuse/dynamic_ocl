use crate::buffer::flags::HostAccess;
use crate::buffer::{Buffer, MemSafe};
use crate::kernel::Kernel;
use crate::raw::{clSetKernelArg, cl_kernel, cl_mem, cl_uint};
use crate::Result;
use libc::size_t;
use std::cell::Cell;
use std::ffi::c_void;
use std::mem::size_of_val;

mod sealed {
    use crate::kernel::Kernel;
    use crate::raw::cl_kernel;
    use crate::safe::kernel::KernelArgList;
    use crate::Result;

    pub trait KernelArgInternal {}
    pub trait KernelArgListInternal {
        fn bind(self, kernel: cl_kernel) -> Result<Kernel<Self>>
        where
            Self: Sized + KernelArgList;
    }
}

/// A trait implemented by types that can be used as an individual kernel
/// argument
pub trait KernelArg: sealed::KernelArgInternal {
    /// Get the data of this kernel argument, as a size and pointer to be passed
    /// to `clSetKernelArg`
    fn as_raw_kernel_arg(&self) -> (size_t, *const c_void);
}

// values can be used as individual kernel args
impl<T: MemSafe> sealed::KernelArgInternal for T {}
impl<T: MemSafe> KernelArg for T {
    fn as_raw_kernel_arg(&self) -> (size_t, *const c_void) {
        (size_of_val(self), self as *const Self as _)
    }
}

// buffers can be used as individual kernel args
impl<H: HostAccess, T: MemSafe> sealed::KernelArgInternal for Buffer<'_, H, T> {}
impl<H: HostAccess, T: MemSafe> KernelArg for Buffer<'_, H, T> {
    fn as_raw_kernel_arg(&self) -> (size_t, *const c_void) {
        (size_of_val(&self.raw()), &self.raw() as *const cl_mem as _)
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
    kernel: cl_kernel,
    index: cl_uint,
    value: Cell<A>,
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
                ptr
            ))?;

            Ok(Self {
                kernel,
                index,
                value: Cell::new(value),
            })
        }
    }

    /// Copy the current value of this argument
    pub fn get(&self) -> K
    where
        K: Copy,
    {
        self.value.get()
    }

    /// Set this kernel argument to a new value, and return the old value
    pub fn replace(&self, new: K) -> Result<K> {
        unsafe {
            let (size, ptr) = new.as_raw_kernel_arg();
            wrap_result!("clSetKernelArg" => clSetKernelArg(
                self.kernel,
                self.index,
                size,
                ptr
            ))?;

            Ok(self.value.replace(new))
        }
    }

    /// Set this kernel argument to a new value
    pub fn set(&self, new: K) -> Result<()> {
        self.replace(new).map(|_| ())
    }
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
