pub mod flags;
mod types;

use crate::raw::*;
use crate::util::sealed::OclInfoInternal;
use flags::*;
use libc::size_t;
use std::ffi::c_void;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
pub use types::*;

// TODO: a buffer could be dropped while other OpenCL objects with implicit
//  references to it (e.g. subbuffers, kernels, etc?) exist, violating safety
//  guarantees?
#[derive(PartialEq, Eq, Hash)]
pub struct Buffer<'a, H: HostAccess, T: MemSafe> {
    _lifetime: PhantomData<&'a ()>,
    _host_access: PhantomData<H>,
    _type: PhantomData<T>,
    pub(crate) handle: cl_mem,
    size: size_t,
}

impl<'a, H: HostAccess, T: MemSafe> Drop for Buffer<'a, H, T> {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = wrap_result!("clReleaseMemObject" => clReleaseMemObject(self.handle)) {
                log::warn!("Error releasing OpenCL mem object: {:?}: {:?}", self, e);
            }
        }
    }
}

impl<'a, H: HostAccess, T: MemSafe> Debug for Buffer<'a, H, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl<'a, H: HostAccess, T: MemSafe> OclInfoInternal for Buffer<'a, H, T> {
    type Param = cl_mem_info;
    const DEBUG_CONTEXT: &'static str = "clGetMemObjectInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        clGetMemObjectInfo(
            self.handle,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl<'a, H: HostAccess, T: MemSafe> Buffer<'a, H, T> {
    pub fn raw(&self) -> cl_mem {
        self.handle
    }

    pub fn rust_size(&self) -> size_t {
        self.size
    }

    info_funcs! {
        pub fn flags(&self) -> BufferFlagsInfo = CL_MEM_FLAGS;
        pub fn size(&self) -> size_t = CL_MEM_SIZE;
        pub fn host_ptr(&self) -> *mut c_void = CL_MEM_HOST_PTR;
        pub fn map_count(&self) -> cl_uint = CL_MEM_MAP_COUNT;
        pub fn reference_count(&self) -> cl_uint = CL_MEM_REFERENCE_COUNT;
        pub fn context_raw(&self) -> cl_context = CL_MEM_CONTEXT;
        pub fn associated_memobject_raw(&self) -> cl_mem = CL_MEM_ASSOCIATED_MEMOBJECT;
        pub fn offset(&self) -> size_t = CL_MEM_OFFSET;
        pub fn uses_svm_pointer(&self) -> bool = CL_MEM_USES_SVM_POINTER;
    }
}
