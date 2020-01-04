use super::flags::*;
use super::Buffer;
use crate::context::Context;
use crate::raw::*;
use crate::Result;
use libc::size_t;
use std::marker::PhantomData;
use std::mem::{size_of, size_of_val};
use std::ptr::null_mut;

/// A trait indicating that a type may be safely stored in an OpenCL buffer.
/// OpenCL buffers allow data to be moved and manipulated in ways that may
/// violate Rust's safety rules. Therefore, this trait may only be implemented
/// for types that have the lifetime `'static`, are sized, have copy semantics,
/// and can be freely  moved. Additionally, the type should be valid for every
/// possible bit pattern,  but this is currently not expressible through Rust's
/// type system.
pub unsafe trait MemSafe: 'static + Sized + Copy + Unpin {
    /// Check whether a given OpenCL C type (e.g. `float` or `ulong`) is
    /// compatible with this Rust type. This will be used when assigning kernel
    /// arguments as a sanity check, but should not be trusted as a guarantee of
    /// correctness.
    fn is_param_type_compatible(c_type: &str) -> bool;
}

unsafe impl MemSafe for cl_char {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "char"
    }
}

unsafe impl MemSafe for cl_uchar {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "uchar"
    }
}

unsafe impl MemSafe for cl_short {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "short"
    }
}

unsafe impl MemSafe for cl_ushort {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "ushort"
    }
}

unsafe impl MemSafe for cl_int {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "int"
    }
}

unsafe impl MemSafe for cl_uint {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "uint"
    }
}

unsafe impl MemSafe for cl_long {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "long"
    }
}

unsafe impl MemSafe for cl_ulong {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "ulong"
    }
}

unsafe impl MemSafe for cl_float {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "float"
    }
}

unsafe impl MemSafe for cl_double {
    fn is_param_type_compatible(c_type: &str) -> bool {
        c_type == "double"
    }
}

#[derive(Clone, Copy)]
pub struct BufferBuilder<
    'c,
    H: HostAccess = HostReadWrite,
    D: DeviceAccess = DeviceReadWrite,
    F: BufferFlags = NoFlags,
> {
    _host_access: PhantomData<H>,
    _device_access: PhantomData<D>,
    _flags: PhantomData<F>,
    context: &'c Context,
}

impl<'c> BufferBuilder<'c> {
    /// Start building a new OpenCL buffer for the given context.
    pub fn new(context: &'c Context) -> Self {
        Self {
            _host_access: PhantomData,
            _device_access: PhantomData,
            _flags: PhantomData,
            context,
        }
    }
}

impl<'c, H: HostAccess, D: DeviceAccess, F: BufferFlags> BufferBuilder<'c, H, D, F> {
    fn update_flags<H2: HostAccess, D2: DeviceAccess, F2: BufferFlags>(
        self,
    ) -> BufferBuilder<'c, H2, D2, F2> {
        BufferBuilder {
            _host_access: PhantomData,
            _device_access: PhantomData,
            _flags: PhantomData,
            context: self.context,
        }
    }

    /// Set the host accessibility for this buffer.
    pub fn host_access<H2: HostAccess>(self) -> BufferBuilder<'c, H2, D, F> {
        self.update_flags()
    }

    /// Set the device accessibility for this buffer.
    pub fn device_access<D2: DeviceAccess>(self) -> BufferBuilder<'c, H, D2, F> {
        self.update_flags()
    }

    /// Set the `CL_MEM_ALLOC_HOST_PTR` flag when creating this buffer.
    pub fn alloc_host_ptr(self) -> BufferBuilder<'c, H, D, AllocHostPtr> {
        self.update_flags()
    }

    fn build<'a, T: MemSafe>(
        self,
        size: size_t,
        host_ptr: *mut T,
        extra_flags: cl_mem_flags,
    ) -> Result<Buffer<'a, H, T>> {
        unsafe {
            let mut err = CL_SUCCESS;

            let handle = clCreateBuffer(
                self.context.raw(),
                H::FLAGS | D::FLAGS | F::FLAGS | extra_flags,
                size,
                host_ptr as _,
                &mut err as _,
            );

            wrap_result!("clCreateBuffer" => err)?;

            Ok(Buffer {
                _lifetime: PhantomData,
                _host_access: PhantomData,
                _type: PhantomData,
                handle,
                size,
            })
        }
    }

    /// Build a buffer, copying initial data from the given slice, with the
    /// `CL_MEM_COPY_HOST_PTR` flag set.
    pub fn build_copying_slice<T: MemSafe>(self, slice: &[T]) -> Result<Buffer<'static, H, T>> {
        self.build(
            size_of_val(slice),
            slice.as_ptr() as *mut _,
            CL_MEM_COPY_HOST_PTR,
        )
    }

    /// Build a buffer with space for `size` elements of type `T`. The initial
    /// contents of the buffer are unspecified.
    pub fn build_with_size<T: MemSafe>(self, size: usize) -> Result<Buffer<'static, H, T>> {
        self.build(size_of::<T>() * size, null_mut(), 0)
    }
}

impl<'c, H: HostAccess, D: DeviceAccess> BufferBuilder<'c, H, D, NoFlags> {
    /// Build a buffer using the given slice for storage, with the
    /// `CL_MEM_USE_HOST_PTR` flag set.
    ///
    /// This type of buffer is still safe thanks to the borrow checker and
    /// `MemSafe` trait, but this also makes the API slightly more difficult to
    /// use. In order to safely access the underlying slice while the buffer
    /// exists, you must map the buffer to "borrow back" a reference to the
    /// slice, which should be a no-op in sane OpenCL implementations.
    pub fn build_using_slice<T: MemSafe>(self, slice: &mut [T]) -> Result<Buffer<H, T>> {
        self.build(size_of_val(slice), slice.as_mut_ptr(), CL_MEM_USE_HOST_PTR)
    }
}

impl Context {
    /// Start building a new OpenCL buffer for this context.
    pub fn buffer_builder(&self) -> BufferBuilder {
        BufferBuilder::new(self)
    }
}

bitfield! {
    /// Flags used to construct a buffer.
    ///
    /// These values cannot be used when constructing a buffer using the safe
    /// API (since traits and type parameters are used instead, to provide
    /// compile-time safety) but they're provided anyways as the output of the
    /// `CL_MEM_FLAGS` info function.
    pub struct BufferFlagsInfo(cl_mem_flags) {
        pub const READ_WRITE = CL_MEM_READ_WRITE;
        pub const WRITE_ONLY = CL_MEM_WRITE_ONLY;
        pub const READ_ONLY = CL_MEM_READ_ONLY;
        pub const USE_HOST_PTR = CL_MEM_USE_HOST_PTR;
        pub const ALLOC_HOST_PTR = CL_MEM_ALLOC_HOST_PTR;
        pub const COPY_HOST_PTR = CL_MEM_COPY_HOST_PTR;
        pub const HOST_WRITE_ONLY = CL_MEM_HOST_WRITE_ONLY;
        pub const HOST_READ_ONLY = CL_MEM_HOST_READ_ONLY;
        pub const HOST_NO_ACCESS = CL_MEM_HOST_NO_ACCESS;
    }
}
