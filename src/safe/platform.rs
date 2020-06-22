//! Platforms
//!
//! An OpenCL platform is an OpenCL driver providing access to specialized
//! hardware. A system can have zero or more platforms, and each platform can
//! provide zero or more devices.

use crate::device::{Device, DeviceType};
use crate::raw::{
    clGetDeviceIDs, clGetPlatformIDs, clGetPlatformInfo, clUnloadPlatformCompiler, cl_platform_id,
    cl_platform_info, cl_ulong,
};
use crate::util::sealed::OclInfoInternal;
use crate::Result;
use std::ffi::c_void;
use std::ffi::CString;
use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::ptr::null_mut;

/// An OpenCL platform
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Platform(pub(crate) cl_platform_id);

unsafe impl Send for Platform {}
unsafe impl Sync for Platform {}

impl Debug for Platform {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.info_fmt(f)
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
        clGetPlatformInfo(
            self.0,
            param_name,
            param_value_size,
            param_value,
            param_value_size_ret,
        )
    }
}

impl Platform {
    /// Get a list of OpenCL platforms available on this system.
    pub fn get_platforms() -> Result<Vec<Platform>> {
        unsafe {
            let mut num_platforms = 0u32;

            wrap_result!("clGetPlatformIDs" => clGetPlatformIDs(
                0,
                null_mut(),
                &mut num_platforms as _
            ))?;

            if num_platforms == 0 {
                return Ok(vec![]);
            }

            let mut ids = vec![null_mut(); num_platforms as usize];

            wrap_result!("clGetPlatformIDs" => clGetPlatformIDs(
                num_platforms,
                ids.as_mut_ptr(),
                &mut num_platforms as _,
            ))?;

            Ok(ids.into_iter().map(Platform).collect())
        }
    }

    /// Get a list of OpenCL devices from this platform.
    pub fn get_devices(self, typ: DeviceType) -> Result<Vec<Device>> {
        unsafe {
            let mut num_devices = 0u32;

            wrap_result!("clGetDeviceIDs" => clGetDeviceIDs(
                self.0,
                typ.raw(),
                0,
                null_mut(),
                &mut num_devices as _
            ))?;

            if num_devices == 0 {
                return Ok(vec![]);
            }

            let mut ids = vec![null_mut(); num_devices as usize];

            wrap_result!("clGetDeviceIDs" => clGetDeviceIDs(
                self.0,
                typ.raw(),
                num_devices,
                ids.as_mut_ptr(),
                &mut num_devices as _,
            ))?;

            Ok(ids.into_iter().map(Device).collect())
        }
    }

    /// Get the raw handle for this platform
    pub fn raw(self) -> cl_platform_id {
        self.0
    }

    /// Wrap the given raw platform handle
    ///
    /// # Safety
    ///
    /// If the given handle is not a valid OpenCL platform ID, behavior is
    /// undefined.
    pub unsafe fn from_raw(handle: cl_platform_id) -> Self {
        Self(handle)
    }

    info_funcs! {
        pub fn profile(&self) -> CString = CL_PLATFORM_PROFILE;
        pub fn version(&self) -> CString = CL_PLATFORM_VERSION;
        pub fn name(&self) -> CString = CL_PLATFORM_NAME;
        pub fn vendor(&self) -> CString = CL_PLATFORM_VENDOR;
        pub fn extensions(&self) -> CString = CL_PLATFORM_EXTENSIONS;
        pub fn host_timer_resolution(&self) -> cl_ulong = CL_PLATFORM_HOST_TIMER_RESOLUTION;
    }

    /// Unload the OpenCL C program compiler for this platform.
    pub fn unload_compiler(self) -> Result<()> {
        unsafe {
            wrap_result!("clUnloadPlatformCompiler" => clUnloadPlatformCompiler(self.raw()))?;
            Ok(())
        }
    }
}
