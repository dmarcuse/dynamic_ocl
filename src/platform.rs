use crate::raw::{cl_platform_id, cl_platform_info};
use crate::util::sealed::OclInfoInternal;
use crate::{OpenCL, Result};
use std::ffi::c_void;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ptr::null_mut;

/// An OpenCL platform
#[derive(Clone)]
pub struct Platform {
    ocl: crate::OpenCL,
    id: cl_platform_id,
}

impl Debug for Platform {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
    }
}

impl PartialEq for Platform {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Platform {}

impl Hash for Platform {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.id as usize)
    }
}

impl OpenCL {
    /// Get a list of OpenCL platforms available on this system.
    ///
    /// This call requires OpenCL 1.0+.
    pub fn get_platforms(&self) -> Result<Vec<Platform>> {
        unsafe {
            let mut num_platforms = 0u32;

            ocl_try!("clGetPlatformIDs" => self.raw().CL10.clGetPlatformIDs(
                0,
                null_mut(),
                &mut num_platforms as _
            ));

            let mut ids = vec![null_mut(); num_platforms as usize];

            ocl_try!("clGetPlatformIDs" => self.raw().CL10.clGetPlatformIDs(
                num_platforms,
                ids.as_mut_ptr(),
                &mut num_platforms as _,
            ));

            Ok(ids
                .into_iter()
                .map(|id| Platform {
                    id,
                    ocl: self.clone(),
                })
                .collect())
        }
    }
}

impl OclInfoInternal for Platform {
    type Param = cl_platform_info;
    const DEBUG_CONTEXT: &'static str = "clGetPlatformInfo";

    unsafe fn raw_info_internal(
        &self,
        param_name: Self::Param,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> i32 {
        self.ocl.raw().CL10.clGetPlatformInfo(
            self.id,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Platform {
    info_funcs! {
        pub fn profile(&self) => self.get_info_string(CL_PLATFORM_PROFILE);
        pub fn version(&self) => self.get_info_string(CL_PLATFORM_VERSION);
        pub fn name(&self) => self.get_info_string(CL_PLATFORM_NAME);
        pub fn vendor(&self) => self.get_info_string(CL_PLATFORM_VENDOR);
        pub fn extensions(&self) => self.get_info_string(CL_PLATFORM_EXTENSIONS);
        pub fn host_timer_resolution(&self) => self.get_info_ulong(CL_PLATFORM_HOST_TIMER_RESOLUTION);
    }
}
